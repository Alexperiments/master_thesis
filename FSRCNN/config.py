import torch

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = False
LOG_REPORT = False
CHECKPOINT = "/m100/home/userexternal/adiana00/Tesi-ML/fsrcnn.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.002#0.000158967
NUM_EPOCHS = 3
BATCH_SIZE = 64
NUM_WORKERS = 8
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1
GPU_NUMBER = 2

NORM_MAX = [0.14, 10.54, 0.028, 0.000044]
NORM_MIN = [0, -14.53, -0.002, 0]

def transform(array, min, max):
    norm = (array-min)/(max-min)
    return torch.from_numpy(norm)

def reverse_transform(norm, min, max):
    array = norm*(max-min) + min
    return array.squeeze(0)
