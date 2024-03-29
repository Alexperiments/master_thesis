import torch
import numpy as np

LOAD_MODEL = True
SAVE_MODEL = False
SAVE_IMG_CHKPNT = True
CHECKPOINT = "esrt.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-3
LR_DECAY = True
NUM_EPOCHS = 1000
BATCH_SIZE = 64
NUM_WORKERS = 2
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

def transform(array, mean=0, std=1):
    max = np.max(array)
    normalized = (array - mean*max)/(std*max)
    return torch.from_numpy(normalized).unsqueeze(0)
