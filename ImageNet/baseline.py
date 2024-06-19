import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
import timm
from sklearn import metrics
import transformers
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
import os
from torchvision import transforms
from torchvision.models import resnet50
import random
from sklearn import metrics
import torch.optim as optim
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from glob import glob
from PIL import ImageFilter, ImageOps
from timm.data import create_transform
from timm.data import Mixup
from datasets import load_dataset
import yaml
from timm.loss import SoftTargetCrossEntropy


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='./config.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    # print(cfg)
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def new_data_aug_generator_DeiT(img_size):
    color_jitter = 0.3
    remove_random_resized_crop = True
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]
   
    if color_jitter is not None and not color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter,color_jitter))
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)

def get_transform(image_size,mean,std):
    return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)])

def build_transform(is_train,input_size):
    print("Applying Random Erase")
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    t = []
    eval_crop_ratio = 0.875
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class IMAGENET100(Dataset):    
    def __init__(self,df,img_dir,transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        #image_id = self.df.iloc[index]['image_id']
        #label = int(self.df.iloc[index]['class'])
        #path = os.path.join(self.img_dir,f'{image_id}.jpeg')
        path = self.df.iloc[index]["file_paths"]
        label = self.df.iloc[index]["class_num"]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)
    
class IMAGENET100_test(Dataset):    
    def __init__(self,img_paths,transform=None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,index):
        image_id = self.img_paths[index].split('/')[-1].split('.')[0]
        path = self.img_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image,image_id

class DeiT_Backbone(nn.Module):
    def __init__(self,image_size,num_classes):
        super(DeiT_Backbone,self).__init__()
        self.encoder = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k',img_size=image_size,pretrained=True,num_classes=0)
        self.fc = nn.Linear(192,num_classes)
        print("Using DieT")
    def forward(self,x):
        x = self.encoder.forward_features(x)
        x = x[:,0,:]
        return self.fc(x)
class BaselineModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048,num_classes)
    def forward(self,x):
        return self.encoder(x)


class Trainer():
    def __init__(self,
                image_size,
                train_dataset,
                val_dataset,
                test_dataset,
                batch_size,
                num_workers,
                device,
                learning_rate,
                epochs,
                model,
                test_output_dir,
                mixup_fn):
        
        self.epochs = epochs
        self.epoch = 0
        self.accelarator = device
        self.model = model
        self.test_output_dir = test_output_dir

        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.test_dataset = test_dataset
        self.image_size = image_size
        self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
        self.train_loader.dataset.transform = new_data_aug_generator_DeiT(self.image_size)
        self.validation_loader = DataLoader(self.val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

        print(f"Length of train loader: {len(self.train_loader)}, validation loader: {(len(self.validation_loader))}, test loader: {len(self.test_loader)}")
    
        self.criterion = nn.CrossEntropyLoss()
        self.lrs = {
            'backbone': learning_rate
        }
        parameters = [{'params': self.model.parameters(),
                        'lr': self.lrs['backbone']},
                        ]
        self.optimizer = optim.AdamW(parameters)
        self.warmup_epochs = 5
        self.decay_factor = 1
        self.steps_per_epoch = len(self.train_loader)//(batch_size)
        if len(self.train_loader) % batch_size != 0:
            self.steps_per_epoch += 1
        self.scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer, self.warmup_epochs*self.steps_per_epoch, self.decay_factor*self.epochs*self.steps_per_epoch)
    
    def get_metrics(self,predictions,actual,isTensor=False):
        if isTensor:
            p = predictions.detach().cpu().numpy()
            a = actual.detach().cpu().numpy()
        else:
            p = predictions
            a = actual
        kappa_score = metrics.cohen_kappa_score(a, p, labels=None, weights='quadratic', sample_weight=None)
        accuracy = metrics.accuracy_score(y_pred=p,y_true=a)
        return {
            "accuracy": accuracy,
            "kappa":  kappa_score
        }

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_step(self):
        self.model.train()
        print("Train Loop!")
        running_loss_train = 0.0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        for images,labels in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)

            labels_metric = labels

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            num_train += labels.shape[0]
            self.optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(images)
                loss = self.criterion(outputs,labels)
                l = loss.item()
                running_loss_train += loss.item()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                #outputs = self.model(images)
                _,preds = torch.max(outputs,1)
                train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
                train_labels = np.concatenate((train_labels,labels_metric.detach().cpu().numpy()))
                #loss = self.criterion(outputs,labels)
                #running_loss_train += loss.item()
                #loss.backward()
                #self.optimizer.step()
        self.lr = self.get_lr(self.optimizer)
        self.scheduler.step()

        train_metrics = self.get_metrics(train_predictions,train_labels)
        print(f"Train Loss: {running_loss_train/num_train}")
        print(f"Train Accuracy Metric: {train_metrics['accuracy']}") 

        return {
                'loss': running_loss_train/num_train,
                'accuracy': train_metrics['accuracy'],
                'kappa': train_metrics['kappa'],
                'lr': self.lr,
            }


    def val_step(self):
        val_predictions = np.array([])
        val_labels = np.array([])
        running_loss_val = 0.0
        num_val = 0
        self.model.eval()
        with torch.no_grad():
            print("Validation Loop!")
            for images,labels in tqdm(self.validation_loader):
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                outputs = self.model(images)
                num_val += labels.shape[0]
                _,preds = torch.max(outputs,1)
                val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                loss = self.criterion(outputs,labels)
                running_loss_val += loss.item()
            val_metrics = self.get_metrics(val_predictions,val_labels)
            print(f"Validation Loss: {running_loss_val/num_val}")
            print(f"Val Accuracy Metric: {val_metrics['accuracy']} ")    
            return {
                'loss': running_loss_val/num_val,
                'accuracy': val_metrics['accuracy'],
                'kappa': val_metrics['kappa'],
            }
    def test_step(self):
        test_image_ids = np.array([])
        test_predictions = np.array([])
        self.model.eval()
        with torch.no_grad():
            print("Test Loop!")
            for images,image_ids in tqdm(self.test_loader):
                images = images.to(self.accelarator)
                outputs = self.model(images)
                _,preds = torch.max(outputs,1)
                test_image_ids = np.concatenate((test_image_ids,image_ids))
                test_predictions = np.concatenate((test_predictions,preds.detach().cpu().numpy()))
            
            pd.DataFrame({
                'image_id':test_image_ids,
                'label':test_predictions.astype(int)
            }).to_csv(os.path.join(self.test_output_dir , 'submission.csv'), index=False)

    def run(self,run_test=True):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0
        wandb.init(project="PatchGD_2_Imagenet_Ablations", name=f"Deit_ImgSize_{IMAGE_SIZE}_baseline")
        for epoch in range(self.epochs):
            print("="*31)
            print(f"{'-'*10} Epoch {epoch+1}/{self.epochs} {'-'*10}")
            train_logs = self.train_step()
            val_logs = self.val_step()
            self.epoch = epoch
            wandb.log({"train_loss": train_logs["loss"],"train_ac": train_logs["accuracy"],"train_kappa": train_logs["kappa"],"lr": train_logs["lr"],"val_loss": val_logs["loss"],"val_acc": val_logs["accuracy"],"val_kappa":val_logs['kappa']})
        
            if val_logs["loss"] < best_validation_loss:
                best_validation_loss = val_logs["loss"]
                os.makedirs(f"/home/akashnyun/PatchGD-git/PatchGD_2.0/ImageNet/{IMAGE_SIZE}_baseline",exist_ok=True)
                torch.save({
                        'model_weights': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.scheduler.state_dict(),
                        'epoch' : epoch+1,
                    }, f"/home/akashnyun/PatchGD-git/PatchGD_2.0/ImageNet/{IMAGE_SIZE}_baseline/best_val_loss.pt")
            if val_logs['accuracy'] > best_validation_accuracy:
                os.makedirs(f"/home/akashnyun/PatchGD-git/PatchGD_2.0/ImageNet/{IMAGE_SIZE}_baseline",exist_ok=True)
                best_validation_accuracy = val_logs['accuracy']
                torch.save({
                        'model_weights': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.scheduler.state_dict(),
                        'epoch' : epoch+1,
                    }, f"/home/akashnyun/PatchGD-git/PatchGD_2.0/ImageNet/{IMAGE_SIZE}_baseline/best_val_acc.pt")
        #    self.test_step()
        print('Best_accuracy',best_validation_accuracy)
        return {
            'best_accuracy':best_validation_accuracy,
            'best_loss': best_validation_loss,
        }    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for training baseline on ImageNet100")
    parser.add_argument('--root_dir',default='/home/akashnyun/imagenet100/',type=str)
    parser.add_argument('--epochs',default=300,type=int)
    parser.add_argument('--batch_size',default=150,type=int)
    parser.add_argument('--image_size',default=384,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--num_classes',default=25,type=int)
    parser.add_argument('--num_workers',default=2,type=int)
    parser.add_argument('--lr',default=1e-5,type=float)
    parser.add_argument('--output_dir',default='./',type=str)
    args = parser.parse_args()
    
    ROOT_DIR = args.root_dir
    IMAGE_SIZE = args.image_size
    EPOCHS = args.epochs
    SEED = args.seed
    BATCH_SIZE = args.batch_size 
    NUM_CLASSES = args.num_classes
    NUM_WORKERS =args.num_workers
    LEARNING_RATE = 0.0005*(BATCH_SIZE/512)#args.lr
    TEST_OUTPUT_DIR = args.output_dir
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    seed_everything(SEED)
    #model = BaselineModel(NUM_CLASSES)
    model = DeiT_Backbone(IMAGE_SIZE, NUM_CLASSES)
    model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    transform = get_transform(IMAGE_SIZE,
                            IMAGENET_DEFAULT_MEAN,
                            IMAGENET_DEFAULT_STD)
    
    

    train_df = pd.read_csv(os.path.join(ROOT_DIR,'train.csv'))
    val_df = pd.read_csv(os.path.join(ROOT_DIR,'val.csv'))
    #train_df = pd.read_csv(os.path.join(ROOT_DIR,'train.csv'))
    #val_df = pd.read_csv(os.path.join(ROOT_DIR,'val.csv'))
    test_files = glob(f'{ROOT_DIR}/test/*')

    mixup_fn = None
    args_,args_text = _parse_args()
    mixup_active = args_.mixup > 0 and args_.cutmix > 0. and args_.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args_.mixup,
            cutmix_alpha=args_.cutmix,
            cutmix_minmax=args_.cutmix_minmax,
            prob=args_.mixup_prob,
            switch_prob=args_.mixup_switch_prob,
            mode=args_.mixup_mode,
            label_smoothing=args_.smoothing,
            num_classes=args_.num_classes
        )
        train_loss_fn = SoftTargetCrossEntropy()
        mixup_fn = Mixup(**mixup_args)

    transform = build_transform(is_train=True,input_size=IMAGE_SIZE)
    train_dataset = IMAGENET100(train_df,os.path.join(ROOT_DIR,"train"),transform)
    transform = build_transform(is_train=False,input_size=IMAGE_SIZE)
    val_dataset = IMAGENET100(val_df,os.path.join(ROOT_DIR,"val"),transform)
    test_dataset = IMAGENET100_test(test_files,transform)

    trainer = Trainer(
        image_size= IMAGE_SIZE,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        model=model,
        test_output_dir=TEST_OUTPUT_DIR,
        mixup_fn=mixup_fn
    )
    
    trainer.run(run_test=True)
    

