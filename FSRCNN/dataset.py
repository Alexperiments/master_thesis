import torch
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config


class MyImageFolder(Dataset):
    def __init__(self):
        super(MyImageFolder, self).__init__()
        self.root_dir = config.TRAIN_FOLDER
        self.image_files_name = sorted(os.listdir(os.path.join(self.root_dir, 'hr')))

    def __len__(self):
        return len(self.image_files_name)

    def __getitem__(self, index):
        file_name = self.image_files_name[index]

        root_and_hr = os.path.join(self.root_dir, "hr")
        hr_array = np.load(os.path.join(root_and_hr, file_name))
        minn = min(hr_array)
        maxx = max(hr_array)
        hr_matrix = config.transform(hr_array, minn, maxx, config.HIGH_RES)

        root_and_lr = os.path.join(self.root_dir, "lr")
        lr_array = np.load(os.path.join(root_and_lr, file_name))
        lr_matrix = config.transform(lr_array, minn, maxx, config.LOW_RES)

        return lr_matrix, hr_matrix


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def test():
    dataset = MyImageFolder()
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

if __name__ == "__main__":
    test()
