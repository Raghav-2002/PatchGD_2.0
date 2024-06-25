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
import copy
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
        self.num_gpus = torch.cuda.device_count()
        self.rank = torch.distributed.get_rank()
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

        images,labels,indices = next(iter(self.train_loader))

        #images = torch.rand(images.shape).to(self.accelarator)
        images = torch.load("image_batch.pt",map_location="cpu").to(self.accelarator)
        #images= images.to(self.accelarator)
        labels = labels.to(self.accelarator)
        batch_size = labels.shape[0]
        num_train = num_train + labels.shape[0]

        L1 = torch.zeros((batch_size,self.latent_dimension,self.num_patches,self.num_patches))
        
        L1 = L1.to(self.accelarator)

        patch_dataset = PatchDataset(images,self.num_patches,self.stride,self.patch_size)
        if DISTRIBUTE_TRAINING_PATCHES:
            print("PATCH PARALLELLISM")
            num_gpus = torch.cuda.device_count()
            patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil((len(patch_dataset)*self.percent_sampling)/num_gpus)),shuffle=False,sampler=DistributedSampler(patch_dataset))
            print("INNER BATCH SIZE", int(math.ceil((len(patch_dataset)*self.percent_sampling)/num_gpus)))
            print("NUMBER OF PATCHES", self.num_patches**2)
        else:
            patch_loader = DataLoader(patch_dataset,batch_size=int(math.ceil(len(patch_dataset)*self.percent_sampling)),shuffle=True)
            print("INNER BATCH SIZE", int(math.ceil((len(patch_dataset)*self.percent_sampling))))
            print("NUMBER OF PATCHES", self.num_patches**2)       
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

        torch.save(L1,f"ZFilled_L1_GPU_RANK_{self.rank}.pt")
        print(f"ZFilled_L1_GPU_RANK_{self.rank}",torch.sum(L1))
        train_loss_sub_epoch = 0
        self.optimizer.zero_grad()
        inner_inter = 0

        if DISTRIBUTE_TRAINING_PATCHES:
            patch_loader.sampler.set_epoch(epoch)

        all_L1 = [torch.zeros_like(out).to(self.accelarator) for _ in range(num_gpus)]
        all_row_id = [torch.zeros_like(row_idx).to(self.accelarator) for _ in range(num_gpus)]
        all_col_id = [torch.zeros_like(col_idx).to(self.accelarator) for _ in range(num_gpus)]

        for inner_iteration, (patches,idxs) in enumerate(patch_loader):
            L1 = L1.detach()
            Lx = copy.deepcopy(L1)
            patches = patches.to(self.accelarator)
            patches = patches.reshape(-1,3,self.patch_size,self.patch_size)
            out = self.model1(patches)
            row_idx = idxs//self.num_patches
            col_idx = idxs%self.num_patches
            out_mod = out.reshape(-1,batch_size, self.latent_dimension)
            out_mod = torch.permute(out_mod,(1,2,0))
            print("l1_row_col",torch.equal(L1[:,:,row_idx,col_idx],out_mod))
            L1[:,:,row_idx,col_idx] = out_mod.to(self.rank)
            print(torch.equal(Lx,L1))
            if DISTRIBUTE_TRAINING_PATCHES:
                all_gather(tensor_list=all_row_id, tensor=torch.tensor(row_idx).to(self.accelarator))
                all_gather(tensor_list=all_col_id, tensor=torch.tensor(col_idx).to(self.accelarator))
                all_gather(tensor_list=all_L1, tensor=torch.tensor(out).to(self.accelarator))
                for index in range(0,len(all_L1)):
                    if torch.equal(all_row_id[index],torch.tensor(row_idx).to(self.accelarator)) and torch.equal(all_col_id[index],torch.tensor(col_idx).to(self.accelarator)):
                        continue
                    out_temp = all_L1[index]
                    out_temp = out_temp.reshape(-1,batch_size, self.latent_dimension)
                    out_temp = torch.permute(out_temp,(1,2,0))
                    L1[:,:,all_row_id[index],all_col_id[index]] = out_temp
            torch.save(L1,f"InnerIteration_L1_GPU_RANK_{self.rank}_inneriteration_{inner_iteration}.pt")
            print(f"InnerIteration_L1_GPU_RANK_{self.rank}_inneriteration_{inner_iteration}",torch.sum(L1))
            outputs = self.model2.forward(L1)
            loss = self.criterion(outputs,labels)
            loss = loss/self.epsilon
            loss = loss
            print("LOSS",loss)
            loss.backward()
            train_loss_sub_epoch = train_loss_sub_epoch + loss.item()
            if (inner_iteration + 1)%self.epsilon==0:
                self.optimizer.step()
                try:
                    print("Grad sum",torch.sum(self.model1.module.encoder.blocks[-1].mlp.fc2.weight.grad))
                    torch.save(self.model1.module.encoder.blocks[-1].mlp.fc2.weight.grad,f"L1_grad_1gpu_{epoch}.pt")
                except:
                    print("Grad None L1")
                try:
                    print("Grad sum L2",torch.sum(self.model2.module.mlp_head.weight.grad))
                    torch.save(self.model2.module.mlp_head.weight.grad,f"L2_grad_1gpu_{epoch}.pt")
                except:
                    print("Grad None L2")
                try:
                    print("Grad sum L2 fl",torch.sum(self.model2.module.transformer.layers[0][0].to_qkv.weight.grad))
                    torch.save(self.model2.module.transformer.layers[0][0].to_qkv.weight.grad,f"L2fl_grad_1gpu_{epoch}.pt")
                except:
                    print("Grad None L2")
                
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

    def run(self):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0
        best_validation_metric = -float('inf')
        for i in range(5):
            print(f"Step {i+1}")
            train_logs = self.train_step(i)
        

def main(rank, world_size,seed,backbone,train_dataset,validation_dataset):
    ddp_setup(rank, world_size)
    trainer = PatchGD_Trainer(seed,backbone,train_dataset,validation_dataset,rank)
    trainer.run()
    destroy_process_group()

def PatchGD_run(seed,backbone,train_dataset,validation_dataset):
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, seed,backbone,train_dataset,validation_dataset), nprocs=world_size)
