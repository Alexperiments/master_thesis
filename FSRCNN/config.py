import torch

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = False
LOG_REPORT = True
CHECKPOINT = "fsrcnn.pth.tar.29_06"
TRAIN_FOLDER = 'train_data_29_06/'
TEST_FOLDER = 'test_data/'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
GPU_NUMBER = 1

LEARNING_RATE = 0.0035  # 0.000158967
NUM_EPOCHS = 400
BATCH_SIZE = 32
BETA1 = 0.9 # 0.957
BETA2 = 0.999 # 0.929

HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

OUTER_CHANNELS = 60
INNER_CHANNELS = 18
MAPS = 8

LR_DECAY_FACTOR = 0.5
DECAY_PATIENCE = 20 # 10

NORM_MAX = [0,0,0,0] #[1.4853e+00, 1.0207e+03, 9.0978e+02, 2.9621e-01] #  [0.14, 10.54, 0.028, 0.000044]
NORM_MIN = [0,0,0,0] #[-4.6654e-03, -3.9330e+02, -2.8211e+01, -1.1846e-06] #  [0, -14.53, -0.002, 0]


def transform(array, min, max):
    norm = (array - min) / (max - min)
    return torch.from_numpy(norm)


def reverse_transform(norm, min, max):
    array = norm * (max - min) + min
    return array.squeeze(0)
