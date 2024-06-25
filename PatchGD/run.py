import argparse
from config import *
from data_utils import *
from models import *
import numpy as np
import warnings
from patch_parallel_test import *
warnings.filterwarnings('ignore')

class DeiT_Backbone(nn.Module):
    def __init__(self,num_classes):
        super(DeiT_Backbone,self).__init__()
        self.encoder = timm.create_model('deit_small_distilled_patch16_224.fb_in1k', img_size=PATCH_SIZE,pretrained=True,num_classes=0)
        
        print("Using DeiT")
    def forward(self,x):
        x = self.encoder.forward_features(x)
        CLS = x[:,0,:]
        return CLS
class Swin_Backbone(nn.Module):
    def __init__(self,num_classes):
        super(Swin_Backbone,self).__init__()
        self.encoder = timm.create_model('timm/swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', img_size=PATCH_SIZE,pretrained=True,num_classes=0)
        
        print("Using Swin")
    def forward(self,x):
        x = self.encoder.forward_features(x)
        CLS = x[:,0,:]
        return CLS

if __name__ == '__main__':
    
    SANITY_CHECK = False
    SANITY_DATA_LEN = None
    if BACKBONE_MODEL == 'DeiT':
        backbone = DeiT_Backbone(NUM_CLASSES)
    elif BACKBONE_MODEL == 'Swin':
        backbone = Swin_Backbone(NUM_CLASSES)
    else:
        raise ValueError(f"Wrong Backbone Name, Expected one from ['DeiT','Swin'] got {BACKBONE_MODEL}")
        
    train_dataset,val_dataset = get_train_val_dataset(TRAIN_CSV_PATH,
                                                    SANITY_CHECK,
                                                    SANITY_DATA_LEN,
                                                    TRAIN_ROOT_DIR,
                                                    VAL_ROOT_DIR,
                                                    IMAGE_SIZE,
                                                    MEAN,
                                                    STD)


    
    PatchGD_run(42,backbone,train_dataset,val_dataset)
