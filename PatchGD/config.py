from datetime import datetime
import torch
import math
from enum import Enum

class model_config(Enum):
    ORIGINAL = 0
    SMALLER = 1
    LARGER = 2
    SMALLER_FEAT = 3
    LARGER_FEAT = 4


now = datetime.now() 
date_time = now.strftime("%d_%m_%Y__%H_%M")

# Distributed Training

DISTRIBUTE_TRAINING_PATCHES = False
DISTRIBUTE_TRAINING_IMAGES = False


# General Parameters
BACKBONE_MODEL = "DeiT"
MONITOR_WANDB = True 
EPOCHS = 100
SAVE_MODELS = True
NUM_CLASSES = 6
SEED = 42
TRAIN_ROOT_DIR = f'/home/akashnyun/pandas_dataset/training_images_512'
VAL_ROOT_DIR = TRAIN_ROOT_DIR
TRAIN_CSV_PATH = f'/home/akashnyun/kfold.csv'
MEAN = [0.9770, 0.9550, 0.9667]
STD = [0.0783, 0.1387, 0.1006]
DECAY_FACTOR = 1
VALIDATION_EVERY = 1
NUM_WORKERS = 1
# Model Hyperparameters
PERCENT_SAMPLING = 0.1
GRAD_ACCUM =  True
BATCH_SIZE = 136
PATCH_SIZE = 128
SCALE_FACTOR = 1 
IMAGE_SIZE = 512*SCALE_FACTOR
WARMUP_EPOCHS = 2
PATCH_BATCHES = math.ceil(1/PERCENT_SAMPLING)
INNER_ITERATION = PATCH_BATCHES
EPSILON = 1
LEARNING_RATE_BACKBONE = 1e-5
LEARNING_RATE_HEAD = 1e-5
STRIDE = PATCH_SIZE
NUM_PATCHES = ((IMAGE_SIZE-PATCH_SIZE)//STRIDE) + 1
NUM_WORKERS = 1
HEAD_TYPE = "MSA" # MSA or CNN
OPTIMIZER_WEIGHT_DECAY = 1e-2
# MSA Parameters
HEADS = 5
DEPTH = 3
MSA_LATENT_DIMENSION = 384
MSA_MLP_LATENT_DIMENSION = MSA_LATENT_DIMENSION*3
# CNN Parameters
CNN_LATENT_DIMENSION = 384
CNN_FEATURE_DIMENSION = 256
CNN_HEAD_TYPE = "original" # original,smaller,larger,smaller_feat,larger_feat


#RUN_NAME = f'{IMAGE_SIZE}_{PATCH_SIZE}-{PERCENT_SAMPLING}-bs-{BATCH_SIZE}-resnet50+head-{FEATURE}-datetime_{date_time}' 
EXPT_NAME = BACKBONE_MODEL+"_ImgSize"+str(IMAGE_SIZE)+"_BS"+str(BATCH_SIZE)+"_IBS"+str(int(((int(IMAGE_SIZE/PATCH_SIZE))**2)*PERCENT_SAMPLING))+"_PS"+str(PATCH_SIZE)+"_"+str(EPSILON)+"IB-Grad_WD-"+str(OPTIMIZER_WEIGHT_DECAY)+"-HeadType-"+str(HEAD_TYPE)
MODEL_SAVE_DIR = f"/home/akashnyun/PatchGD-git/PatchGD_2.0/PatchGD/checkpoints_{BACKBONE_MODEL}_{IMAGE_SIZE}_{BATCH_SIZE}_PatchGD_{HEAD_TYPE}/"





 
