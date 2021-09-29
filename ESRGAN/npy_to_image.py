import torch
import os, glob
import numpy as np
from PIL import Image

npy_folder = 'upscaled/'
png_folder = 'upscaled/png/'

os.system(f"mkdir -p {png_folder}")

files = glob.glob(os.path.join(npy_folder, "*.npy"))

for i, file in enumerate(files):
    sample = np.load(file)
    matrix_image = np.uint8(sample*255)
    img = Image.fromarray(matrix_image, mode='L')
    img.save(png_folder +
            file.replace(npy_folder, '').replace('.npy', '.png'))
