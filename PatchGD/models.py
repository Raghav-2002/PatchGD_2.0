import torch.nn as nn
from torchvision.models import resnet50
from config import model_config
import numpy as np
import torch
import timm
from config import *
from transformers import DeiTConfig, DeiTModel
from transformers import AutoConfig, AutoModel

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from config import *



class CNN_Block(nn.Module):
    def __init__(self,num_layers_sr,latent_dim,feature_dim,num_classes,num_patches):
        super(CNN_Block,self).__init__()
        self.expected_dim = (2,latent_dim,num_patches,num_patches)
        self.same_layers,self.reduction_layers = num_layers_sr #(same,reduction)
        self.layers = []
        self.feature_dim = feature_dim
        for _ in range(self.same_layers):
            self.layers.append(
                nn.Sequential(
                nn.Conv2d(latent_dim,self.feature_dim,3,1,1), 
                nn.ReLU(),
                nn.BatchNorm2d(self.feature_dim),
                nn.Dropout2d(p=0.2))
            )
        d = self.feature_dim
        if self.same_layers == 0:
            d = latent_dim
        for _ in range(self.reduction_layers):
            self.layers.append(
                nn.Sequential(
                nn.Conv2d(d,self.feature_dim,3,2,1), 
                nn.ReLU(),
                nn.BatchNorm2d(self.feature_dim),
                nn.Dropout2d(p=0.2))
            )
        self.layers = nn.Sequential(*self.layers)
        flatten_dim = self.get_final_out_dimension(self.expected_dim)
        self.linear = nn.Linear(flatten_dim,num_classes)

    def get_output_shape(self, model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

    def get_final_out_dimension(self,shape):
        s = shape
        s = self.get_output_shape(self.layers,s)
        return np.prod(list(s[1:]))

    def forward(self,x):
        for l in self.layers:
            x = l(x)
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)
        return x



def get_head(config_type,latent_dimension,feature_dimension,num_classes,num_patches):
    num_layers_sr = (1,3)
    
    if config_type == model_config.SMALLER:
        num_layers_sr = (0,3)
    elif config_type == model_config.LARGER:
        num_layers_sr = (1,4)
    elif config_type == model_config.SMALLER_FEAT:
        feaure_dimension = 128
    elif config_type == model_config.LARGER_FEAT:
        feaure_dimension = 512
    return CNN_Block(num_layers_sr=num_layers_sr,
                    latent_dim=latent_dimension,
                    feature_dim=feaure_dimension,
                    num_classes=num_classes,
                    num_patches=num_patches)
    

def CNN_HEAD(name,latent_dimension,feature_dimension,num_classes,num_patches):
    configurations = {'original':model_config.ORIGINAL,
                    'smaller': model_config.SMALLER,
                    'larger': model_config.LARGER,
                    'smaller_feat': model_config.SMALLER_FEAT,
                    'larger_feat': model_config.LARGER_FEAT}
    return get_head(configurations[name],latent_dimension,feature_dimension,num_classes,num_patches)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
 

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class MSA_HEAD(nn.Module):
    def __init__(self, *,accelarator,num_patches,num_classes, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding = torch.tensor(getPositionEncoding(seq_len=num_patches+1, d=dim))[None,:].to(accelarator)
        self.dropout = nn.Dropout(emb_dropout)
        self.feature_dim =  dim
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self,features):
        features = torch.permute(features,(0,2,3,1))
        features = features.reshape(features.shape[0],-1,self.feature_dim)
        b, n, _ = features.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, features), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:,0]
        x = self.to_latent(x)
        return self.mlp_head(x)