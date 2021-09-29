import os, glob
import numpy as np

source = "dataset/"
target = "data/"

os.system("mkdir -p numpy_data/ numpy_data/lr numpy_data/hr")
os.system("mkdir -p "+target+" "+target+"lr/ "+target+"hr/")
os.system("mv "+source+"*_20.npy numpy_data/lr/")
os.system("mv "+source+"*_80.npy numpy_data/hr/")
os.system("rm -f -- numpy_data/lr/grid_20.npy")
os.system("rm -f -- numpy_data/hr/grid_80.npy")
os.system("rename 's/_20//' numpy_data/lr/*.npy")
os.system("rename 's/_80//' numpy_data/hr/*.npy")
os.system("rm -fr -- "+source)

for res in ['lr', 'hr']:
    npy_folder = 'numpy_data/' + res
    files = glob.glob(os.path.join(npy_folder, "*.npy"))

    for i, file in enumerate(files):
        sample = np.load(file)
        dim = int(np.sqrt(len(sample)))
        matrix_image = sample.reshape(dim, dim)
        np.save(target + res +
                file.replace(npy_folder, ''), matrix_image)

os.system("rm -fr -- numpy_data")
