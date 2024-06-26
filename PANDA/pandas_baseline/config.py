from datetime import datetime
import os
import torch

DEVICE_ID = 1 ######################################
now = datetime.now() 
date_time = now.strftime("%d_%m_%Y__%H_%M")
MAIN_RUN = True ######################################

MONITOR_WANDB = True ######################################
SAVE_MODELS =  MONITOR_WANDB
BASELINE = True
SANITY_CHECK = False
EPOCHS = 100
LEARNING_RATE = 1e-4
ACCELARATOR = f'cuda:6' if torch.cuda.is_available() else 'cpu'
SCALE_FACTOR = 1 ######################################
IMAGE_SIZE = int(SCALE_FACTOR * 512)
BATCH_SIZE = 44 ######################################
FEATURE = '' ######################################
MEMORY = 4 ######################################
RUN_NAME = f'{DEVICE_ID}-{IMAGE_SIZE}-{BATCH_SIZE}-resnet50-baseline-{MEMORY}GB-{FEATURE}-datetime_{date_time}' ######################################

NUM_CLASSES = 6
SEED = 42
LEARNING_RATE_BACKBONE = LEARNING_RATE
LEARNING_RATE_HEAD = LEARNING_RATE
WARMUP_EPOCHS = 2
NUM_WORKERS = 4
TRAIN_ROOT_DIR = f'/home/raghavmagazine/pandas_dataset/training_images_512'
VAL_ROOT_DIR = TRAIN_ROOT_DIR
TRAIN_CSV_PATH = f'/home/raghavmagazine/kfold.csv'
MEAN = [0.9770, 0.9550, 0.9667]
STD = [0.0783, 0.1387, 0.1006]
SANITY_DATA_LEN = None
MODEL_SAVE_DIR = "/home/raghavmagazine/PatchGD-main/pandas_baseline/Pre_Trained_PANDA"
#f"../{'models_icml' if MAIN_RUN else 'models'}/{'sanity' if SANITY_CHECK else 'runs'}/{RUN_NAME}"
EXPERIMENT = "pandas-shared-runs-icml-rebuttal" if not SANITY_CHECK else 'pandas-sanity-gowreesh'
DECAY_FACTOR = 2
RANDOM_PATCH_SIZE=64
RUN_NAME_WANDB = "Baseline_PreTrained_PANDA_ImgSize512_lr_1e-4_WD"
