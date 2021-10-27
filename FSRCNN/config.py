import torch

LOAD_MODEL = False
SAVE_MODEL = True
SAVE_IMG_CHKPNT = True
CHECKPOINT = "fsrcnn.pth.tar"
TRAIN_FOLDER = 'train_data/'
TEST_FOLDER = 'test_data/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0025#0.000158967
NUM_EPOCHS = 4000
BATCH_SIZE = 512
NUM_WORKERS = 4
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

def transform(array, min, max):
    norm = (array-min)/(max-min)
    return torch.from_numpy(norm).unsqueeze(0)

def reverse_transform(norm, min, max):
    array = norm*(max-min) + min
    return array.squeeze(0).squeeze(0)
