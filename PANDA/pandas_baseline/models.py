
import torch.nn as nn
import timm
from torchvision.models import resnet50
from config import *
from transformers import DeiTConfig, DeiTModel
class Backbone(nn.Module):
    def __init__(self,num_classes):
        super(Backbone,self).__init__()
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(2048,num_classes)
        print("Using ResNet")
    def forward(self,x):
        return self.encoder(x)

def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook


class DeiT_Backbone(nn.Module):
    def __init__(self,num_classes):
        super(DeiT_Backbone,self).__init__()
        #self.encoder = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k',img_size=RANDOM_PATCH_SIZE,pretrained=True,num_classes=0)
        self.encoder = timm.create_model('deit_tiny_distilled_patch16_224.fb_in1k',img_size=512,pretrained=True,num_classes=0)
        self.fc = nn.Linear(192,num_classes)
        print("Using DieT")
    def forward(self,x):
        x = self.encoder.forward_features(x)
        x = x[:,0,:]
        return self.fc(x)

