import numpy as np
from copy import deepcopy
np.set_printoptions(precision=5, suppress=True)
import tensorflow as tf

import argparse
from typing import Any, Generator, Mapping, Tuple
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax._src import transform
from jax import jit, grad, vmap
from jax.tree_util import tree_structure
import pickle
from collections import Counter


OptState = Any
Batch = Mapping[str, np.ndarray]

BUFFER_SIZE = 10

def make_ds(features, labels):
  ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
  ds = ds.shuffle(BUFFER_SIZE).repeat()
  return ds


def read_fmri_data(root_dir,years):


    def process_year(y):


        X_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}_test/X*.npy").as_numpy_iterator())
        # X_2 = list(tf.data.Dataset.list_files(root_dir + "/V2_test/X*.npy").as_numpy_iterator())
        # X_3 = list(tf.data.Dataset.list_files(root_dir + "/V3_test/X*.npy").as_numpy_iterator())

        y_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}_test/y*.npy").as_numpy_iterator())
        # y_2 = list(tf.data.Dataset.list_files(root_dir + "/V2_test/y*.npy").as_numpy_iterator())
        # y_3 = list(tf.data.Dataset.list_files(root_dir + "/V3_test/y*.npy").as_numpy_iterator())

        V1 = {'X': {}, 'y': {}}
        # V2 = {'X': {}, 'y': {}}
        # V3 = {'X': {}, 'y': {}}

        for i in range(len(X_1)):
            try:
                index = int(X_1[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
                V1['X'][index - 1] = np.load(X_1[i], mmap_mode='r').astype(np.float32)
            except:
                print(f"An exception occurred at index {i} of V{y}: {X_1[i]}")
                continue

            try:
                index = int(y_1[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
                V1['y'][index - 1] = np.load(y_1[i], mmap_mode='r').astype(np.float32)
            except:
                print(f"An exception occurred at index {i} of V{y}: {y_1[i]}")
                continue


        X = []
        y = []


        for key in sorted(V1['X']):
            try:
                X.append(V1['X'][key])
                y.append(V1['y'][key])
            except:
                print(f"An exception occurred for key {key} in V{y}.")
                print(V1['X'][key].shape)
                print(V1['y'][key].shape)
                continue


        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)


        # normalize data
        X = (X - X.mean()) / X.std()


        # binary classes 1 or 5
        X_pos = X[y == 4]
        X_neg = X[y == 5]
        y_pos = y[y == 4]
        y_neg = y[y == 5]
        #X = X[(y == 4) | (y == 5)]
        #y = y[(y == 4) | (y == 5)]

        print(f"X pos shape: {X_pos.shape}")
        print(f"y pos shape: {y_pos.shape}")
        print(f"X neg shape: {X_neg.shape}")
        print(f"y neg shape: {y_neg.shape}")

        return X_pos, y_pos, X_neg, y_neg

    datasets = []
    pos_ds_l = []
    neg_ds_l = []
    for y in years:
        print(f"Processing year {y} data...")
        X_pos, y_pos, X_neg, y_neg = process_year(y)
        pos_ds = make_ds(X_pos, y_pos)
        neg_ds = make_ds(X_neg, y_neg)
        pos_ds_l.append(pos_ds)
        neg_ds_l.append(neg_ds)

        # for features, label in pos_ds.take(1):
        #     print("Features:\n", features.numpy())
        #     print()
        #     print("Label: ", label.numpy())
        #datasets.append(tf.data.Dataset.from_tensor_slices({'X': X, 'y': Y}))

    pos_ds = pos_ds_l[0].concatenate(pos_ds_l[1]) if len(pos_ds_l) > 1 else pos_ds_l[0]
    neg_ds = neg_ds_l[0].concatenate(neg_ds_l[1]) if len(neg_ds_l) > 1 else neg_ds_l[0]
    resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    resampled_ds = resampled_ds.batch(batch_size).prefetch(2)

    return resampled_ds

def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
    seed: int,
    dataset
    ) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = dataset.cache().repeat()
    # if is_training:
    #     ds = ds.shuffle(10 * batch_size, seed)
    ds = ds.batch(batch_size)

    return iter(ds.as_numpy_iterator())
#    return iter(balanced_ds.take(batch_size).as_numpy_iterator())


if __name__ == "__main__":


    root_dir = "/Users/sean/Projects/pgm-fmri/data"
    batch_size=8
    seed=0
    all = []
    # n_envs = 1
    ds_train_envs = []
    print(f"Batch size: {batch_size}")
    years = [1,2]
    train_ds = read_fmri_data(root_dir,years)
    test_ds = read_fmri_data(root_dir, [3])

    # ds = datasets[0].concatenate(datasets[1])
    # for m in range(n_envs):

    ds = load_dataset("train", is_training=True, batch_size=batch_size, dataset=train_ds, seed=seed)

    test_ds = load_dataset("test", is_training=False, batch_size=batch_size, dataset=test_ds, seed=seed)
