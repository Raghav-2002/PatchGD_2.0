import os
DEVICE_IDS = 1
#os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_IDS)
from config import *
from data_utils import *
from train_utils import *
from utils import *
from models import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    DEVICE_ID = DEVICE_IDS
    MONITOR_WANDB = False
    SCALE_FACTOR = 1
    IMAGE_SIZE = int(SCALE_FACTOR * 512)
    BATCH_SIZE = 64 ######################################
    FEATURE = '' ######################################
    MEMORY = 4 ######################################
    
    
    EXPERIMENT = "pandas-shared-runs-icml-rebuttal" if not SANITY_CHECK else 'pandas-sanity-gowreesh-2'
    SAVE_MODELS =  True
    NUM_EXPERIMENTS = 5
    RUN_NAME = f'{DEVICE_ID}-{IMAGE_SIZE}-{BATCH_SIZE}-resnet50-baseline-{MEMORY}GB-{FEATURE}-datetime_{date_time}' ######################################

    if SAVE_MODELS:
        os.makedirs(MODEL_SAVE_DIR,exist_ok=True)

    if MONITOR_WANDB:
        run = wandb.init(project=EXPERIMENT, entity="aiiscaiisc078", reinit=True)
        wandb.run.name = RUN_NAME
        wandb.run.save()

    train_dataset,val_dataset = get_train_val_dataset(TRAIN_CSV_PATH,
                                                    SANITY_CHECK,
                                                    SANITY_DATA_LEN,
                                                    TRAIN_ROOT_DIR,
                                                    VAL_ROOT_DIR,
                                                    IMAGE_SIZE,
                                                    MEAN,
                                                    STD)

    best_accuracies = []
    best_metrics = []

    print(DEVICE_ID)
    print(RUN_NAME)

    seeds = np.random.randint(10,10000,NUM_EXPERIMENTS)
    for run_number, seed in enumerate([42]):
        print(f"Run Number:{run_number}, seed:{seed}")
        if MONITOR_WANDB:
            wandb.log({
                'run_number': run_number,
                'seed': seed,
            })
        
        trainer = Trainer(seed,
                    run_number,
                    train_dataset,
                    val_dataset,
                    BATCH_SIZE,
                    NUM_WORKERS,
                    NUM_CLASSES,
                    ACCELARATOR,
                    RUN_NAME,
                    LEARNING_RATE,
                    EPOCHS,
                    WARMUP_EPOCHS,
                    DECAY_FACTOR,
                    MONITOR_WANDB,
                    SAVE_MODELS,
                    MODEL_SAVE_DIR)
        logs = trainer.run()
        best_accuracies.append(logs['best_accuracy'])
        best_metrics.append(logs['best_metric'])

        del trainer
    print('mean_accuracies',np.mean(best_accuracies))
    print('std_accuracies',np.std(best_accuracies))
    print('mean_kappa',np.mean(best_metrics))
    print('std_kappa',np.std(best_metrics))
    if MONITOR_WANDB:
        print({
            'mean_accuracies': np.mean(best_accuracies),
            'std_accuracies': np.std(best_accuracies),
            'mean_kappa': np.mean(best_metrics),
            'std_kappa': np.std(best_metrics),
        })

        wandb.log({
            'mean_accuracies': np.mean(best_accuracies),
            'std_accuracies': np.std(best_accuracies),
            'mean_kappa': np.mean(best_metrics),
            'std_kappa': np.std(best_metrics),
        })


                
