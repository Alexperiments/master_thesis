import torch
import numpy as np

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = False
CHECKPOINT = "ffcnn.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE =  0.005#0.000158967
NUM_EPOCHS = 500
BATCH_SIZE = 1024
NUM_WORKERS = 4
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

def transform(array, min=3, max=10):
    return torch.from_numpy(array).unsqueeze(0)

def reverse_transform(array, min=3, max=10):
    return array.squeeze(0).squeeze(0)
