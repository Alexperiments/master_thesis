import torch
import numpy as np

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = True
CHECKPOINT = "fsrcnn.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE =  0.002#0.000158967
NUM_EPOCHS = 2000
BATCH_SIZE = 32
NUM_WORKERS = 2
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

def transform(array, mean=0, std=1):
    max = np.max(array)
    normalized = (array - mean*max)/(std*max)
    return torch.from_numpy(normalized).unsqueeze(0)