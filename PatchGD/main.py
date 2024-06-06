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
        self.scaler = torch.cuda.amp.GradScaler()
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
        self.model1.train()
        self.model2.train()
        if self.accelarator == 0:
            print("Train Loop!")
        running_loss_train = 0.0
        train_correct = 0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])

        if DISTRIBUTE_TRAINING_IMAGES:
            self.train_loader.sampler.set_epoch(epoch)


        for images,labels,indices in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)
            batch_size = labels.shape[0]
            num_train = num_train + labels.shape[0]

            L1 = torch.zeros((batch_size,self.latent_dimension,self.num_patches,self.num_patches))
            
            L1 = L1.to(self.accelarator)

            patch_dataset = PatchDataset(images,self.num_patches,self.stride,self.patch_size)
            if DISTRIBUTE_TRAINING_PATCHES:
                num_gpus = torch.cuda.device_count()
                patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil((len(patch_dataset)*self.percent_sampling)/num_gpus)),shuffle=False,sampler=DistributedSampler(patch_dataset))
            else:
                patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil(len(patch_dataset)*self.percent_sampling)),shuffle=True)

            with torch.no_grad():
                for patches, idxs in patch_loader:
                    patches = patches.to(self.accelarator)
                    patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                    out = self.model1(patches)
                    row_idx = idxs//self.num_patches
                    col_idx = idxs%self.num_patches
                    break

            num_gpus = torch.cuda.device_count()
            all_L1 = [torch.zeros_like(out).to(self.accelarator) for _ in range(num_gpus)]
            all_row_id = [torch.zeros_like(row_idx).to(self.accelarator) for _ in range(num_gpus)]
            all_col_id = [torch.zeros_like(col_idx).to(self.accelarator) for _ in range(num_gpus)]
            

            with torch.no_grad():
                for patches, idxs in patch_loader:
                    patches = patches.to(self.accelarator)
                    patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                    out = self.model1(patches)
                    row_idx = idxs//self.num_patches
                    col_idx = idxs%self.num_patches

                    if DISTRIBUTE_TRAINING_PATCHES:
                        all_gather(tensor_list=all_row_id, tensor=torch.tensor(row_idx).to(self.accelarator))
                        all_gather(tensor_list=all_col_id, tensor=torch.tensor(col_idx).to(self.accelarator))
                        all_gather(tensor_list=all_L1, tensor=torch.tensor(out).to(self.accelarator))
                        for index in range(0,len(all_L1)):
                            out_temp = all_L1[index]
                            out_temp = out_temp.reshape(-1,batch_size, self.latent_dimension)
                            out_temp = torch.permute(out_temp,(1,2,0))
                            L1[:,:,all_row_id[index],all_col_id[index]] = out_temp 
                    else:
                        out = out.reshape(-1,batch_size, self.latent_dimension)
                        out = torch.permute(out,(1,2,0))
                        L1[:,:,row_idx,col_idx] = out


            train_loss_sub_epoch = 0
            self.optimizer.zero_grad()
            inner_inter = 0

            if DISTRIBUTE_TRAINING_PATCHES:
                patch_loader.sampler.set_epoch(epoch)

            all_L1 = [torch.zeros_like(out).to(self.accelarator) for _ in range(num_gpus)]
            all_row_id = [torch.zeros_like(row_idx).to(self.accelarator) for _ in range(num_gpus)]
            all_col_id = [torch.zeros_like(col_idx).to(self.accelarator) for _ in range(num_gpus)]

            for inner_iteration, (patches,idxs) in enumerate(patch_loader):
                with torch.autocast(device_type = "cuda", dtype = torch.float16,enabled = FP16):
                    L1 = L1.detach()
                    patches = patches.to(self.accelarator)
                    patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                    out = self.model1(patches)
                    row_idx = idxs//self.num_patches
                    col_idx = idxs%self.num_patches

                    if DISTRIBUTE_TRAINING_PATCHES:
                        all_gather(tensor_list=all_row_id, tensor=torch.tensor(row_idx).to(self.accelarator))
                        all_gather(tensor_list=all_col_id, tensor=torch.tensor(col_idx).to(self.accelarator))
                        all_gather(tensor_list=all_L1, tensor=torch.tensor(out).to(self.accelarator))
                        for index in range(0,len(all_L1)):
                            out_temp = all_L1[index]
                            out_temp = out_temp.reshape(-1,batch_size, self.latent_dimension)
                            out_temp = torch.permute(out_temp,(1,2,0))
                            L1[:,:,all_row_id[index],all_col_id[index]] = out_temp
                    else:
                        out = out.reshape(-1,batch_size, self.latent_dimension)
                        out = torch.permute(out,(1,2,0))
                        L1[:,:,row_idx,col_idx] = out


                    outputs = self.model2.forward(L1)
                    loss = self.criterion(outputs,labels)
                    loss = loss/self.epsilon
                if FP16==True:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                train_loss_sub_epoch = train_loss_sub_epoch + loss.item()

                if (inner_iteration + 1)%self.epsilon==0:
                    if FP16 == True:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    inner_inter =  inner_inter + 1
                if inner_iteration + 1 >= self.inner_iteration:
                    break
                    
            self.scheduler.step()
            running_loss_train = running_loss_train + train_loss_sub_epoch
            with torch.no_grad():
                _,preds = torch.max(outputs,1)
                correct = (preds == labels).sum().item()
                train_correct = train_correct + correct

                train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
                train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

            self.lr = self.get_lr(self.optimizer)


        # Collecting results accross GPUs for multi-GPU Training 
        num_gpus = torch.cuda.device_count()
        all_labels = [torch.zeros(len(train_labels),dtype=torch.float64).to(self.accelarator) for _ in range(num_gpus)]
        all_predictions = [torch.zeros(len(train_predictions),dtype=torch.float64).to(self.accelarator) for _ in range(num_gpus)]
        all_gather(tensor_list=all_labels, tensor=torch.tensor(train_labels).to(self.accelarator))
        all_gather(tensor_list=all_predictions, tensor=torch.tensor(train_predictions).to(self.accelarator))
        train_predictions = torch.cat(all_predictions).detach().cpu().numpy()
        train_labels = torch.cat(all_labels).detach().cpu().numpy()

        train_metrics = self.get_metrics(train_predictions,train_labels)

        if self.accelarator == 0:
            print(f"Train Loss: {running_loss_train/num_train} Train Accuracy Metric: {train_metrics['accuracy']} Train Kappa Metric: {train_metrics['kappa']}")    
        
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
        self.model2.eval()
        with torch.no_grad():
            if self.accelarator == 0:
                print("Validation Loop!")
            for images,labels,indices in tqdm(self.validation_loader):
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                batch_size = labels.shape[0]
                
                L1 = torch.zeros((batch_size,self.latent_dimension,self.num_patches,self.num_patches))
                L1 = L1.to(self.accelarator)

                patch_dataset = PatchDataset(images,self.num_patches,self.stride,self.patch_size)
                patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil(len(patch_dataset)*self.percent_sampling)),shuffle=True)

                num_patch_batches = 0
                with torch.no_grad():
                    for patches, idxs in patch_loader:
                        num_patch_batches = num_patch_batches + 1
                        patches = patches.to(self.accelarator)
                        patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
                        out = self.model1(patches)
                        out = out.reshape(-1,batch_size, self.latent_dimension)
                        out = torch.permute(out,(1,2,0))
                        row_idx = idxs//self.num_patches
                        col_idx = idxs%self.num_patches
                        L1[:,:,row_idx,col_idx] = out

                outputs = self.model2.forward(L1)
                num_val = num_val + labels.shape[0]
                _,preds = torch.max(outputs,1)
                val_correct = val_correct + (preds == labels).sum().item()
                correct = (preds == labels).sum().item()

                val_metrics_step = self.get_metrics(preds,labels,True)
                val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                loss = self.criterion(outputs,labels)
                l = loss.item()
                running_loss_val = running_loss_val + loss.item()
            
            val_metrics = self.get_metrics(val_predictions,val_labels)
            if self.accelarator == 0:
                print(f"Validation Loss: {running_loss_val/num_val} Val Accuracy Metric: {val_metrics['accuracy']} Val Kappa Metric: {val_metrics['kappa']}")    
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
