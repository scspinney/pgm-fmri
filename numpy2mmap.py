import numpy as np
import os
import glob
from tempfile import mkdtemp



def process_data(root_dir, year):
    """

    :param root_dir: Directory where the data is stored at the top level
    :param year: integer indicating the year to process
    :return: 
    """


    X_1 = glob.glob(os.path.join(root_dir,f"V{year}/X*.npy"))

    y_1 = glob.glob(os.path.join(root_dir,f"V{year}/y*.npy"))

    V1 = {'X': {}, 'y': {}}


    for i in range(len(X_1)):
        try:
            index = int(X_1[i].split('/')[-1].split('.')[0].split('_')[-1])
            V1['X'][index - 1] = np.load(X_1[i], mmap_mode='r').astype(np.float32)
        except Exception as e:
            print(f"An exception occurred at index {i} of V{year}: {X_1[i]}")
            print(f"{e}")
            continue

        try:
            index = int(y_1[i].split('/')[-1].split('.')[0].split('_')[-1])
            V1['y'][index - 1] = np.load(y_1[i], mmap_mode='r').astype(np.float32)
        except Exception as e:
            print(f"An exception occurred at index {i} of V{year}: {y_1[i]}")
            print(f"{e}")
            continue

    V1_X = []

    V1_y = []


    for key in sorted(V1['X']):
        try:
            V1_X.append(V1['X'][key])
            V1_y.append(V1['y'][key])
        except:
            print(f"An exception occurred for key {key} in V{year}.")
            print(V1['X'][key].shape)
            print(V1['y'][key].shape)
            continue


    V1_X = np.concatenate(V1_X, axis=0)

    V1_y = np.concatenate(V1_y, axis=0)

    # remove class 0 or go_scan
    V1_X = V1_X[V1_y != 0]

    V1_y = V1_y[V1_y != 0]

    V1_X = V1_X.reshape(-1, 53, 63, 52)

    V1_y = V1_y.reshape(-1)

    # normalize data
    V1_X = (V1_X - V1_X.mean()) / V1_X.std()

    # save as memmap
    filename = os.path.join(root_dir,f"V{year}/V{year}_X_mmap.dat")
    print(f"Saving X to : {filename}")
    V1_X_mmap = np.memmap(filename, dtype='float32', mode='w+', shape=V1_X.shape)
    V1_X_mmap[:] = V1_X[:]
    V1_X_mmap.flush()

    filename = os.path.join(root_dir,f"V{year}/V{year}_y_mmap.dat")
    print(f"Saving y to : {filename}")
    V1_y_mmap = np.memmap(filename, dtype='float32', mode='w+', shape=V1_y.shape)
    V1_y_mmap[:] = V1_y[:]
    V1_y_mmap.flush()

    print("Done.")



if __name__ == "__main__":

    years = [1,2,3]
    root_dir = "/home/spinney/scratch/fmri"
    for y in years:
        print(f"Processing year {y}")
        process_data(root_dir,y)
