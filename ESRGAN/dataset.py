import torch
from tqdm import tqdm
import time
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
from PIL import Image
import cv2


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        self.image_file_name = sorted(os.listdir(os.path.join(self.root_dir, 'hr')))

    def __len__(self):
        return len(self.image_file_name)

    def __getitem__(self, index):
        file_name = self.image_file_name[index]

        root_and_lr = os.path.join(self.root_dir, "lr")
        low_res = cv2.imread(os.path.join(root_and_lr, file_name))
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
        low_res = config.transform(image=low_res)["image"]

        root_and_hr = os.path.join(self.root_dir, "hr")
        high_res = cv2.imread(os.path.join(root_and_hr, file_name))
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        high_res = config.transform(image=high_res)["image"]

        return low_res, high_res


def test():
    dataset = MyImageFolder(root_dir="data/")
    loader = DataLoader(dataset, batch_size=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
