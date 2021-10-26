import torch
import numpy as np

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = False
CHECKPOINT = "fsrcnn.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE =  0.0025#0.000158967
LR_DECAY = True
NUM_EPOCHS = 1000
BATCH_SIZE = 512
NUM_WORKERS = 4
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

def transform(array, min, max, size):
    norm = (array-min)/(max-min)
    matrix = norm.reshape(size, size)
    return torch.from_numpy(matrix).unsqueeze(0)

def reverse_transform(norm, min, max):
    array = norm*(max-min) + min
    return array.squeeze(0).squeeze(0)
