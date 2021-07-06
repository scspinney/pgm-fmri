import numpy
import os
import glob
import shutil

years = [1,2,3]

root_dir = "/scratch/spinney/fmri"
sample_percentage = 0.3

for y in years:
    cur_dir = os.path.join(root_dir,f"V{y}")
    X_files = sorted(glob.glob(os.path.join(cur_dir,"X*.npy")))
    y_files = sorted(glob.glob(os.path.join(cur_dir, "y*.npy")))

    total = len(X_files)
    max_index = int(total*sample_percentage)
    X_files = X_files[:max_index]
    y_files = y_files[:max_index]
    file_names = X_files + y_files
    for file_name in file_names:
        fname = file_name.split('/')[-1]
        new_name = os.path.join(root_dir,f"V{y}_test",fname)
        print(f"Copying file {file_name} to {new_name}")
        shutil.copy(file_name, new_name)
