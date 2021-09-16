import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen_noVGG.pth.tar"
CHECKPOINT_DISC = "disc_noVGG.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 2
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

transform = A.Compose(
    [
     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
     ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
