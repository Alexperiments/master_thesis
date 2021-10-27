import os, glob
import numpy as np
import random

source = "dataset/"
temp = "temp_folder/"
train = "train_data/"
test = "test_data/"

os.system(f"mkdir -p {temp} {temp}lr {temp}hr")
os.system(f"mkdir -p {train} {train}lr/ {train}hr/")
os.system(f"mkdir -p {test} {test}lr/ {test}hr/")
os.system(f"cp {source}*_20.npy {temp}lr/")
os.system(f"cp {source}*_80.npy {temp}hr/")
os.system(f"rm -f -- {temp}lr/grid_20.npy")
os.system(f"rm -f -- {temp}hr/grid_80.npy")
os.system(f"rename 's/_20//' {temp}lr/*.npy")
os.system(f"rename 's/_80//' {temp}hr/*.npy")
#os.system("rm -fr -- "+source)

try:
    npy_folder = temp + 'lr/'
    files = os.listdir(npy_folder)
    random.shuffle(files)
    len_files = len(files)
    train_size = int(len_files*0.8)


    for res in ['lr/', 'hr/']:
        for file in files[:train_size]:
            sample = np.load(temp + res + file)
            dim = int(np.sqrt(len(sample)))
            matrix_image = sample.reshape(dim, dim)
            np.save(
                train + res + file,
                matrix_image
            )

        for file in files[train_size:]:
            sample = np.load(temp + res + file)
            dim = int(np.sqrt(len(sample)))
            matrix_image = sample.reshape(dim, dim)
            np.save(
                test + res + file,
                matrix_image
            )

except:
    os.system("rm -fr -- "+train)
    os.system("rm -fr -- "+test)
    os.system("rm -fr -- "+temp)
    raise

os.system(f"rm -fr -- {temp}")
os.system(f"rm -fr -- {source}")
