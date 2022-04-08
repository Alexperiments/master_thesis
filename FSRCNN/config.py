import torch

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = False
LOG_REPORT = True
CHECKPOINT = "fsrcnn.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
GPU_NUMBER = 1

LEARNING_RATE = 0.002#0.000158967
NUM_EPOCHS = 21
BATCH_SIZE = 256

HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 4
OUTER_CHANNELS = 56
INNER_CHANNELS = 12
MAPS = 5

LR_DECAY_FACTOR = 0.5
DECAY_PATIENCE = 10

NORM_MAX = [0.14, 10.54, 0.028, 0.000044]
NORM_MIN = [0, -14.53, -0.002, 0]

def transform(array, min, max):
    norm = (array-min)/(max-min)
    return torch.from_numpy(norm)

def reverse_transform(norm, min, max):
    array = norm*(max-min) + min
    return array.squeeze(0)
