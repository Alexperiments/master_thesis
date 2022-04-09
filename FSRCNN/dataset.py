import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

import config


class MyImageFolder(Dataset):
    def __init__(self):
        super(MyImageFolder, self).__init__()
        print("HGGEEY")
        os.getcwd()
        self.root_dir = config.TRAIN_FOLDER
        path = os.path.join(self.root_dir, 'hr')
        self.image_files_name = sorted(os.listdir(path))

    def __len__(self):
        return len(self.image_files_name)

    def __getitem__(self, index):
        file_name = self.image_files_name[index]
        map = [0,1,2,3]
        delta_max = config.NORM_MAX.copy()
        delta_min = config.NORM_MIN.copy()
        #del delta_min[1]
        #del delta_max[1]
        delta_min = np.swapaxes(np.array([[delta_min]]), 0, 2)
        delta_max = np.swapaxes(np.array([[delta_max]]), 0, 2)

        root_and_lr = os.path.join(self.root_dir, "lr")
        lr_array = np.float32(np.load(os.path.join(root_and_lr, file_name))[map])
        maxx = np.float32(np.amax(lr_array, axis=(1, 2), keepdims=True) + delta_max)
        minn = np.float32(np.amin(lr_array, axis=(1, 2), keepdims=True) + delta_min)
        lr_matrix = config.transform(lr_array, minn, maxx)

        root_and_hr = os.path.join(self.root_dir, "hr")
        hr_array = np.float32(np.load(os.path.join(root_and_hr, file_name))[map])
        hr_matrix = config.transform(hr_array, minn, maxx)

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


class SingleExampleDataFolder(Dataset):
    def __init__(self, line_velocity=False):
        super(SingleExampleDataFolder, self).__init__()
        self.root_dir = config.TRAIN_FOLDER
        path = os.path.join(self.root_dir, 'hr')
        self.image_files_name = sorted(os.listdir(path))
        self.flag = line_velocity

    def __len__(self):
        return len(self.image_files_name)

    def __getitem__(self, index):
        file_name = self.image_files_name[index]

        delta_max = config.NORM_MAX.copy()
        delta_min = config.NORM_MIN.copy()
        maps = [0, 1, 2, 3]

        if not self.flag:
            del delta_min[1]
            del delta_max[1]
            del maps[1]

        delta_min = np.swapaxes(np.array([[delta_min]]), 0, 2)
        delta_max = np.swapaxes(np.array([[delta_max]]), 0, 2)

        root_and_lr = os.path.join(self.root_dir, "lr")
        lr_array = np.float32(np.load(os.path.join(root_and_lr, file_name))[maps])
        maxx = np.float32(np.amax(lr_array, axis=(1, 2), keepdims=True) + delta_max)
        minn = np.float32(np.amin(lr_array, axis=(1, 2), keepdims=True) + delta_min)
        lr_matrix = config.transform(lr_array, minn, maxx)

        root_and_hr = os.path.join(self.root_dir, "hr")
        hr_array = np.float32(np.load(os.path.join(root_and_hr, file_name))[maps])
        hr_matrix = config.transform(hr_array, minn, maxx)

        return lr_matrix.unsqueeze(0).view(-1, 1, config.LOW_RES, config.LOW_RES), \
               hr_matrix.unsqueeze(0).view(-1, 1, config.HIGH_RES, config.HIGH_RES)


def test():
    dataset = MyImageFolder()
    loader = MultiEpochsDataLoader(dataset, batch_size=config.BATCH_SIZE)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
