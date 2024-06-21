from sklearn import metrics
from data_utils import *
from models import *
import torch.optim as optim
import transformers
from tqdm import tqdm
import wandb
import torch.nn as nn
import os
from config import *
import math
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_gather
import os

def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PatchGD_Trainer():
    def __init__(self,seed,backbone,train_dataset,validation_dataset,accelarator):

        seed_everything(seed)

        
        self.epochs = EPOCHS
        self.num_classes = NUM_CLASSES
        self.epoch = 0
        self.monitor_wandb = MONITOR_WANDB
        self.save_models = SAVE_MODELS
        self.model_save_dir = MODEL_SAVE_DIR
        self.accelarator = accelarator
        self.batch_size = BATCH_SIZE
        self.head = CNN_HEAD_TYPE
        self.inner_iteration = INNER_ITERATION
        self.grad_accumulation = GRAD_ACCUM
        self.epsilon = EPSILON
        self.num_patches = NUM_PATCHES
        self.stride = STRIDE
        self.patch_size = PATCH_SIZE
        self.percent_sampling = PERCENT_SAMPLING
        self.train_dataset = train_dataset 
        self.val_dataset = validation_dataset 
        if DISTRIBUTE_TRAINING_IMAGES:
            self.train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,sampler=DistributedSampler(train_dataset))
        else:
            self.train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)

        self.validation_loader = DataLoader(validation_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)
        
        if self.monitor_wandb and self.accelarator == 0:
            wandb.init(project="PatchGD-2.0", name=EXPT_NAME)
    
        if self.save_models:
            os.makedirs(self.model_save_dir,exist_ok=True)


        backbone = backbone.to("cuda")
        self.model1 = DDP(backbone, device_ids=[self.accelarator])
        for param in self.model1.parameters():
            param.requires_grad = True

        if HEAD_TYPE == "MSA":
            head_arch = MSA_HEAD(accelarator=self.accelarator,num_patches=int(IMAGE_SIZE//PATCH_SIZE)**2,num_classes=NUM_CLASSES,dim=MSA_LATENT_DIMENSION,depth=DEPTH,heads=HEADS,mlp_dim=MSA_MLP_LATENT_DIMENSION,dropout=0.1,emb_dropout = 0.1)
            head_arch = head_arch.to("cuda")
            self.model2 = DDP(head_arch, device_ids=[self.accelarator])
            self.latent_dimension = MSA_LATENT_DIMENSION
        
        elif HEAD_TYPE == "CNN":
            head_arch = CNN_HEAD(name = self.head,latent_dimension=CNN_LATENT_DIMENSION,feature_dimension=CNN_FEATURE_DIMENSION,num_classes=NUM_CLASSES,num_patches=self.num_patches)
            head_arch = head_arch.to("cuda")
            self.model2 = DDP(head_arch, device_ids=[self.accelarator])
            self.latent_dimension = CNN_LATENT_DIMENSION

        else:
            print("Invalid Head")

        for param in self.model2.parameters():
            param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()
        self.lrs = {
            'head': LEARNING_RATE_HEAD,
            'backbone': LEARNING_RATE_BACKBONE
        }
        parameters = [{'params': self.model1.parameters(),
                    'lr': self.lrs['backbone']},
                    {'params': self.model2.parameters(),
                    'lr': self.lrs['head']}]
        self.optimizer = optim.Adam(parameters,weight_decay=OPTIMIZER_WEIGHT_DECAY)
        steps_per_epoch = len(self.train_dataset)//(self.batch_size)

        if len(self.train_dataset)%self.batch_size!=0:
            steps_per_epoch = steps_per_epoch + 1
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,WARMUP_EPOCHS*steps_per_epoch,DECAY_FACTOR*self.epochs*steps_per_epoch)
        
    
    def get_metrics(self,predictions,actual,isTensor=False):
        if isTensor:
            p = predictions.detach().cpu().numpy()
            a = actual.detach().cpu().numpy()
        else:
            p = predictions
            a = actual
        kappa_score = metrics.cohen_kappa_score(a, p, labels=None, weights= 'quadratic', sample_weight=None)
        accuracy = metrics.accuracy_score(y_pred=p,y_true=a)
        return {
            "kappa":  kappa_score,
            "accuracy": accuracy
        }

    def get_lr(self,optimizer):
        lrs = []
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
        return lrs

    def train_step(self,epoch):
        self.train_loader.sampler.set_epoch(epoch)
        self.model1.train()
        print("Train Loop!")
        running_loss_train = 0.0
        train_correct = 0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])
        for images,labels in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)
            batch_size = labels.shape[0]
            num_train += labels.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model1(images)
            # if torch.isnan(outputs).any():
            #     print("output has nan")
            _,preds = torch.max(outputs,1)
            train_correct += (preds == labels).sum().item()
            correct = (preds == labels).sum().item()

            train_metrics_step = self.get_metrics(preds,labels,True)
            train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
            train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

            loss = self.criterion(outputs,labels)
            l = loss.item()
            running_loss_train += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.lr = self.get_lr(self.optimizer)
            if self.monitor_wandb and self.accelarator==0:
                wandb.log({'lr':self.lr,"train_loss_step":l/batch_size,'epoch':self.epoch,'train_accuracy_step_metric':train_metrics_step['accuracy'],'train_kappa_step_metric':train_metrics_step['kappa']})
        train_metrics = self.get_metrics(train_predictions,train_labels)
        print(f"Train Loss: {running_loss_train/num_train} Train Accuracy: {train_correct/num_train}")
        print(f"Train Accuracy Metric: {train_metrics['accuracy']} Train Kappa Metric: {train_metrics['kappa']}")    
        return {
                'loss': running_loss_train/num_train,
                'accuracy': train_metrics['accuracy'],
                'kappa': train_metrics['kappa']
            }


    def val_step(self):
        val_predictions = np.array([])
        val_labels = np.array([])
        running_loss_val = 0.0
        val_correct  = 0
        num_val = 0
        self.model1.eval()
        with torch.no_grad():
            print("Validation Loop!")
            for images,labels in tqdm(self.validation_loader):
    
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                batch_size = labels.shape[0]

                outputs = self.model1(images)
                # if torch.isnan(outputs).any():
                #     print("L1 has nan")
                num_val += labels.shape[0]
                _,preds = torch.max(outputs,1)
                val_correct += (preds == labels).sum().item()
                correct = (preds == labels).sum().item()

                val_metrics_step = self.get_metrics(preds,labels,True)
                val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                loss = self.criterion(outputs,labels)
                l = loss.item()
                running_loss_val += loss.item()
                if self.monitor_wandb and self.accelarator==0:
                    wandb.log({'lr':self.lr,"val_loss_step":l/batch_size,"epoch":self.epoch,'val_accuracy_step_metric':val_metrics_step['accuracy'],'val_kappa_step_metric':val_metrics_step['kappa']})
            
            val_metrics = self.get_metrics(val_predictions,val_labels)
            print(f"Validation Loss: {running_loss_val/num_val} Validation Accuracy: {val_correct/num_val}")
            print(f"Val Accuracy Metric: {val_metrics['accuracy']} Val Kappa Metric: {val_metrics['kappa']}")    
            return {
                'loss': running_loss_val/num_val,
                'accuracy': val_metrics['accuracy'],
                'kappa': val_metrics['kappa']
            }

    def run(self):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0
        best_validation_metric = -float('inf')
        for epoch in range(self.epochs):
            if self.accelarator == 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
            train_logs = self.train_step(epoch)
            val_logs = self.val_step()
            self.epoch = epoch

            if self.accelarator == 0:
                if val_logs["loss"] < best_validation_loss:
                    best_validation_loss = val_logs["loss"]
                    if self.save_models:
                        torch.save({
                            'model1_weights': self.model1.module.state_dict(),
                            'model2_weights': self.model2.module.state_dict(),
                            'optimizer_state': self.optimizer.state_dict(),
                            'scheduler_state': self.scheduler.state_dict(),
                            'epoch' : epoch+1,
                        }, f"{self.model_save_dir}/best_val_loss.pt")

                if val_logs['accuracy'] > best_validation_accuracy:
                    best_validation_accuracy = val_logs['accuracy']
                    if self.save_models:
                        torch.save({
                            'model1_weights': self.model1.module.state_dict(),
                            'model2_weights': self.model2.module.state_dict(),
                            'optimizer_state': self.optimizer.state_dict(),
                            'scheduler_state': self.scheduler.state_dict(),
                            'epoch' : epoch+1,
                        }, f"{self.model_save_dir}/best_val_accuracy.pt")

                if val_logs['kappa'] > best_validation_metric:
                    best_validation_metric = val_logs['kappa']
                    if self.save_models:
                        torch.save({
                            'model1_weights': self.model1.module.state_dict(),
                            'model2_weights': self.model2.module.state_dict(),
                            'optimizer_state': self.optimizer.state_dict(),
                            'scheduler_state': self.scheduler.state_dict(),
                            'epoch' : epoch+1,
                        }, f"{self.model_save_dir}/best_val_metric.pt")

                if self.monitor_wandb:
                    wandb.log({"training_loss": train_logs['loss'],  
                    "validation_loss": val_logs['loss'], 
                    'training_accuracy_metric': train_logs['accuracy'],
                    'training_kappa_metric': train_logs['kappa'],
                    'validation_accuracy_metric': val_logs['accuracy'],
                    'validation_kappa_metrics': val_logs['kappa'],
                    'epoch':self.epoch,
                    'best_loss':best_validation_loss,
                    'best_accuracy':best_validation_accuracy,
                    'best_metric': best_validation_metric,
                    'learning_rate': self.lr[0]})
                    
                if self.save_models:
                    torch.save({
                    'model1_weights': self.model1.module.state_dict(),
                    'model2_weights': self.model2.module.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'epoch' : epoch+1,
                    }, f"{self.model_save_dir}/last_epoch.pt")
        
        return {
            'best_accuracy':best_validation_accuracy,
            'best_loss': best_validation_loss,
            'best_metric': best_validation_metric
        }

def main(rank, world_size,seed,backbone,train_dataset,validation_dataset):
    ddp_setup(rank, world_size)
    trainer = PatchGD_Trainer(seed,backbone,train_dataset,validation_dataset,rank)
    trainer.run()
    destroy_process_group()

def PatchGD_run(seed,backbone,train_dataset,validation_dataset):
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, seed,backbone,train_dataset,validation_dataset), nprocs=world_size)
