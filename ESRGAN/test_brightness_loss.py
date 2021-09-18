import numpy as np
import os
import cv2
import config

path_lr = os.path.join("data/lr/", "0.png")
path_hr = os.path.join("data/hr/", "0.png")

lr_image = cv2.imread(path_lr)
lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY if config.IMG_CHANNELS==1 else cv2.COLOR_BGR2RGB)

hr_image = cv2.imread(path_hr)
hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2GRAY if config.IMG_CHANNELS==1 else cv2.COLOR_BGR2RGB)

i, j = 0, 16
N=1
print(hr_image[i*4:(i+N)*4,j*4:(j+N)*4])
print(lr_image[i:i+N,j:j+N])
