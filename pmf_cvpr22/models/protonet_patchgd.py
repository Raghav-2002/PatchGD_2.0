import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from copy import deepcopy
from tqdm import tqdm
from timm.utils import accuracy
from .protonet import ProtoNet
from .utils import trunc_normal_, DiffAugment
from torch.utils.data import Dataset, DataLoader
import math
from config import *

class PatchDataset(Dataset):
    def __init__(self,images,num_patches,stride,patch_size):
        self.images = images
        self.num_patches = num_patches
        self.stride = stride
        self.patch_size = patch_size
    def __len__(self):
        return self.num_patches ** 2
    def __getitem__(self,choice):
        i = choice%self.num_patches
        j = choice//self.num_patches
        return self.images[:,:,self.stride*i:self.stride*i+self.patch_size,self.stride*j:self.stride*j+self.patch_size], choice


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@torch.jit.script
def entropy_loss(x):
    return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()


def unique_indices(x):
    """
    Ref: https://github.com/rusty1s/pytorch_unique
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, perm



class ProtoNet_Finetune_PatchGD(ProtoNet):
    def __init__(self, backbone, num_iters=50, lr=5e-2, aug_prob=0.9,
                 aug_types=['color', 'translation']):
        super().__init__(backbone)
        self.num_iters = num_iters
        self.lr = lr
        self.aug_types = aug_types
        self.aug_prob = aug_prob

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        state_dict = self.backbone.state_dict()
        self.backbone_state = deepcopy(state_dict)

    def forward(self, supp_x, supp_y, x, model_head):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        # reset backbone state
        self.backbone.load_state_dict(self.backbone_state, strict=True)

        #if self.lr == 0:
        #    return super().forward(supp_x, supp_y, x)

        B, nSupp, C, H, W = supp_x.shape
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
        device = x.device

        criterion = nn.CrossEntropyLoss()
        supp_x = supp_x.view(-1, C, H, W)
        x = x.view(-1, C, H, W)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp
        supp_y = supp_y.view(-1)


        #PatchGD
        num_patches = NUM_PATCHES
        stride = PATCH_SIZE
        patch_size = PATCH_SIZE
        percent_sampling = PERCENT_SAMPLING
        latent_dimension = LATENT_DIM
        max_inner_iteration = MAX_INNER_ITERATION
        epsilon = EPSILON

        # create optimizer
        parameters = [{'params': self.backbone.parameters(),'lr': self.lr},
                    {'params': model_head.parameters(),'lr': HEAD_LR}]
        opt = torch.optim.Adam(parameters,
                               betas=(0.9, 0.999),
                               weight_decay=0.)

        def single_step_patchgd(z, optimizer,mode=True):
            '''
            z = Aug(supp_x) or x
            '''

            with torch.set_grad_enabled(mode):
                # recalculate prototypes from supp_x with updated backbone
                #supp_f = self.backbone.forward(supp_x)

                #PatchGD
                supp_x_patch_dataset = PatchDataset(supp_x,num_patches,stride,patch_size)
                supp_x_patch_loader = DataLoader(supp_x_patch_dataset,batch_size=int(math.ceil(len(supp_x_patch_dataset)*percent_sampling)),shuffle=True)
                batch_size_supp_x = supp_x.size()[0]

                L1_supp_x = torch.zeros((batch_size_supp_x,latent_dimension,num_patches,num_patches))
                L1_supp_x  = L1_supp_x.to(device)

                batch_size = z.size()[0]
                z_patch_dataset = PatchDataset(z,num_patches,stride,patch_size)
                z_patch_loader = DataLoader(z_patch_dataset,batch_size=int(math.ceil(len(z_patch_dataset)*percent_sampling)),shuffle=True)

                L1 = torch.zeros((batch_size,latent_dimension,num_patches,num_patches))
                L1 = L1.to(device)

                with torch.no_grad():
                    for patches_supp_x, idxs_supp_x in supp_x_patch_loader:
                        patches_supp_x = patches_supp_x.to(device)
                        patches_supp_x = patches_supp_x.reshape(-1,3,patch_size,patch_size)
                        out_supp_x = self.backbone.forward(patches_supp_x)
                        out_supp_x = out_supp_x.reshape(-1,batch_size_supp_x, latent_dimension)
                        out_supp_x = torch.permute(out_supp_x,(1,2,0))
                        row_idx_supp_x = idxs_supp_x//num_patches
                        col_idx_supp_x = idxs_supp_x%num_patches
                        L1_supp_x[:,:,row_idx_supp_x,col_idx_supp_x] = out_supp_x

                        for patches, idxs in z_patch_loader:
                            patches = patches.to(device)
                            patches = patches.reshape(-1,3,patch_size,patch_size)
                            out = self.backbone.forward(patches)
                            out = out.reshape(-1,batch_size, latent_dimension)
                            out = torch.permute(out,(1,2,0))
                            row_idx = idxs//num_patches
                            col_idx = idxs%num_patches
                            L1[:,:,row_idx,col_idx] = out

                optimizer.zero_grad()

                for inner_iteration, ((patches_supp_x,idxs_supp_x), (patches,idxs)) in enumerate(zip(supp_x_patch_loader,z_patch_loader)):
                    L1_supp_x  = L1_supp_x.detach()
                    patches_supp_x = patches_supp_x.to(device)
                    patches_supp_x = patches_supp_x.reshape(-1,3,patch_size,patch_size)
                    out_supp_x = self.backbone.forward(patches_supp_x)
                    out_supp_x = out_supp_x.reshape(-1,batch_size_supp_x, latent_dimension)
                    out_supp_x = torch.permute(out_supp_x,(1,2,0))
                    row_idx_supp_x = idxs_supp_x//num_patches
                    col_idx_supp_x = idxs_supp_x%num_patches
                    L1_supp_x[:,:,row_idx_supp_x,col_idx_supp_x] = out_supp_x
                    supp_f = model_head.forward(L1_supp_x)
                    
                    
                    supp_f = supp_f.view(B, nSupp, -1)
                    prototypes = torch.bmm(supp_y_1hot.float(), supp_f) # B, nC, d
                    prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

                    # compute feature for z
                    # feat = self.backbone.forward(z)
                        
                    L1 = L1.detach()
                    patches = patches.to(device)
                    patches = patches.reshape(-1,3,patch_size,patch_size)
                    out = self.backbone.forward(patches)
                    out = out.reshape(-1,batch_size, latent_dimension)
                    out = torch.permute(out,(1,2,0))
                    row_idx = idxs//num_patches
                    col_idx = idxs%num_patches
                    L1[:,:,row_idx,col_idx] = out
                    feat = model_head.forward(L1)
                    ######---------------------------######
                    feat = feat.view(B, z.shape[0], -1) # B, nQry, d
                    # classification
                    logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
                    prototypes = prototypes.detach()
                    feat = feat.detach()
                    loss = None
                    if mode:
                        loss_1 = criterion(logits.view(B*nSupp, -1), supp_y)
                        loss = loss_1/epsilon
                    #####-----------------------------#####

                    loss.backward()
                    
                    if (inner_iteration + 1)%epsilon==0:
                        optimizer.step()
                        optimizer.zero_grad()
                    if inner_iteration + 1 >= max_inner_iteration:
                        break
                    
            return loss_1

        def single_step(z, optimizer,mode=True):
            '''
            z = Aug(supp_x) or x
            '''

            with torch.set_grad_enabled(mode):
                # recalculate prototypes from supp_x with updated backbone
                #supp_f = self.backbone.forward(supp_x)

                #PatchGD
                supp_x_patch_dataset = PatchDataset(supp_x,num_patches,stride,patch_size)
                supp_x_patch_loader = DataLoader(supp_x_patch_dataset,batch_size=int(math.ceil(len(supp_x_patch_dataset)*percent_sampling)),shuffle=True)
                batch_size_supp_x = supp_x.size()[0]

                L1_supp_x = torch.zeros((batch_size_supp_x,latent_dimension,num_patches,num_patches))
                L1_supp_x  = L1_supp_x.to(device)

                with torch.no_grad():
                    for patches_supp_x, idxs_supp_x in supp_x_patch_loader:
                        patches_supp_x = patches_supp_x.to(device)
                        patches_supp_x = patches_supp_x.reshape(-1,3,patch_size,patch_size)
                        out_supp_x = self.backbone.forward(patches_supp_x)
                        out_supp_x = out_supp_x.reshape(-1,batch_size_supp_x, latent_dimension)
                        out_supp_x = torch.permute(out_supp_x,(1,2,0))
                        row_idx_supp_x = idxs_supp_x//num_patches
                        col_idx_supp_x = idxs_supp_x%num_patches
                        L1_supp_x[:,:,row_idx_supp_x,col_idx_supp_x] = out_supp_x

                supp_f = model_head.forward(L1_supp_x)
                supp_f = supp_f.view(B, nSupp, -1)
                prototypes = torch.bmm(supp_y_1hot.float(), supp_f) # B, nC, d
                prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0

                # compute feature for z
                # feat = self.backbone.forward(z)

                batch_size = z.size()[0]
                z_patch_dataset = PatchDataset(z,num_patches,stride,patch_size)
                z_patch_loader = DataLoader(z_patch_dataset,batch_size=int(math.ceil(len(z_patch_dataset)*percent_sampling)),shuffle=True)

                L1 = torch.zeros((batch_size,latent_dimension,num_patches,num_patches))
                L1 = L1.to(device)

                #PatchGD
                with torch.no_grad():
                    for patches, idxs in z_patch_loader:
                        patches = patches.to(device)
                        patches = patches.reshape(-1,3,patch_size,patch_size)
                        out = self.backbone.forward(patches)
                        out = out.reshape(-1,batch_size, latent_dimension)
                        out = torch.permute(out,(1,2,0))
                        row_idx = idxs//num_patches
                        col_idx = idxs%num_patches
                        L1[:,:,row_idx,col_idx] = out
                feat = model_head.forward(L1)
                feat = feat.view(B, z.shape[0], -1) # B, nQry, d
                # classification
                logits = self.cos_classifier(prototypes, feat) # B, nQry, nC

            return logits

        # main loop
        pbar = tqdm(range(self.num_iters)) if is_main_process() else range(self.num_iters)
        for i in pbar:
            z = DiffAugment(supp_x, self.aug_types, self.aug_prob, detach=True)
            loss = single_step_patchgd(z, opt,True)
            #loss.backward()
            #opt.step()
            if is_main_process():
                pbar.set_description(f'BB_lr{self.lr}, HD_lr{HEAD_LR}, nSupp{nSupp}, nQry{x.shape[0]}: loss = {loss.item()}')

        logits = single_step(x, False)
        return logits


