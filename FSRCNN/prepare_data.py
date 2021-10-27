import os, glob
import numpy as np
import random

source = "dataset/"
target = "train_data/"

os.system(f"mkdir -p {target} {target}lr {target}hr")
os.system(f"cp {source}parameters.txt {target}")

for res, pix in zip(['lr', 'hr'], ['20', '80']):
    os.system(f"cp {source}grid_{pix}.npy {target}")
    os.system(f"cp {source}*_{pix}.npy {target}{res}/")
    os.system(f"rm -f -- {target}{res}/grid_{pix}.npy")
    os.system(f"rename 's/_{pix}//' {target}{res}/*.npy")

print("copied")
N = len(os.listdir(f"{target}lr/"))
names = np.arange(0, N)
for res, dim in zip(['lr', 'hr'], [20, 80]):
    for name in names:
        array = np.load(f"{target}{res}/{name}.npy")
        np.save(f"{target}{res}/{name}.npy", array.reshape(dim, dim))
