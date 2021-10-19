import torch
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import config



import matplotlib.pyplot as plt

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
        lr_array = np.load(os.path.join(root_and_lr, file_name))

        root_and_hr = os.path.join(self.root_dir, "hr")
        hr_array = np.load(os.path.join(root_and_hr, file_name))
        hr_matrix = config.transform(hr_array)

        return lr_array, hr_matrix


def extract_features():
    inputs = pd.read_csv("min_max_20.txt", sep='\t')
    inputs = inputs.astype({"ID": int})
    inputs.set_index('ID', inplace=True)

    #min-max normalization
    inputs=(inputs-inputs.min())/(inputs.max()-inputs.min())

    path = config.TRAIN_FOLDER + 'lr/'
    for i, row in inputs.iterrows():
        obj_path = path + str(i) + '.npy'
        np.save(obj_path, row)


def test():
    dataset = MyImageFolder(root_dir=config.TRAIN_FOLDER)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
    #extract_features()
