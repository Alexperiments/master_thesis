import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 2
HIGH_RES = 80
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 1

transform = A.Compose(
    [
     A.Normalize(mean=[0.5 for _ in range(IMG_CHANNELS)], std=[0.5 for _ in range(IMG_CHANNELS)]),
     ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0 for _ in range(IMG_CHANNELS)], std=[1 for _ in range(IMG_CHANNELS)]),
        ToTensorV2(),
    ]
)
