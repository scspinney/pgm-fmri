from os import pread
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
# from optax._src import transform, base
from jax import jit, grad, vmap
from jax.tree_util import tree_structure
import pickle
from collections import Counter
from sklearn.metrics import classification_report
from haiku.data_structures import to_mutable_dict


OptState = Any
Batch = Mapping[str, np.ndarray]
BUFFER_SIZE = 10



def storeData(object, file_name, root_dir):
    with open(root_dir+file_name, 'wb') as f:
        pickle.dump(object, f)					 
        f.close() 

def loadData(file_name, root_dir): 
    with open(root_dir+file_name, 'rb') as f:
        db = pickle.load(f) 
        f.close()
        return db


def make_ds(features, labels):
  ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
  ds = ds.shuffle(BUFFER_SIZE)
  return ds


def read_fmri_data(root_dir,years):


    def process_year(y):


        X_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}/X/X*.npy").as_numpy_iterator())
        # X_2 = list(tf.data.Dataset.list_files(root_dir + "/V2_test/X*.npy").as_numpy_iterator())
        # X_3 = list(tf.data.Dataset.list_files(root_dir + "/V3_test/X*.npy").as_numpy_iterator())

        y_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}/y/y*.npy").as_numpy_iterator())
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
    #resampled_ds = resampled_ds.batch(batch_size)

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
    ds = dataset.cache()
    # if is_training:
    #     ds = ds.shuffle(10 * batch_size, seed)
    ds = ds.batch(batch_size)

    return iter(ds.as_numpy_iterator())

def get_preds(ds, params, n_classes=2, normalizer=255.):


    preds_per_batch = []
    true_ys = []

    def net_fn(batch) -> jnp.ndarray:
    
        x = jnp.array(batch, jnp.float32) / normalizer
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(n_classes),
        ])
        return mlp(x)


    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))

    # Training loss (cross-entropy).
    def preds(params: hk.Params, batch) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        # print(params)
        # m = to_mutable_dict(params)
        net_out = net.apply(params[0], batch)
        preds = jnp.argmax(net_out, axis=-1)
        return jax.device_get(preds)

    for step, (x_batch, y_batch) in enumerate(ds):
        print(f"batch number {step+1} ")
        true_ys.append(y_batch.astype('int32'))
        current_batch_preds = preds(params=params, batch=x_batch)
        preds_per_batch.append(current_batch_preds)

    true_ys = np.concatenate(true_ys, axis=0)
    preds_per_batch = np.concatenate(preds_per_batch, axis=0)
    return classification_report(true_ys, preds_per_batch)
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--outname", type=str)

    args = parser.parse_args()

    if args.path == None:
        root_dir = "/Users/sean/Projects/pgm-fmri/data"
        batch_size = 2
        outname = "testing-01"
        seed = 33
        n_classes = 2

    else:
        root_dir = args.path
        batch_size = args.batch_size
        outname = args.outname
        seed = args.seed
        n_classes = args.n_classes

    print("the path is ", root_dir)

    tf.config.experimental.set_visible_devices([], "GPU")

    print(f"Batch size: {batch_size}")

    test_years = [3]

    test_ds = read_fmri_data(root_dir, test_years)

    data = loadData('fmri/fmriall_hps_B32_S0_ILC', root_dir=root_dir)

    for d in data:
        test_ds_it = load_dataset("test", is_training=False, batch_size=batch_size, seed=seed, dataset=test_ds)
        print(f"thresh: {d['thresh']}, l1: {d['l1']}, l2: {d['l2']}")
        d['scores'] = [get_preds(test_ds_it, d['params'], n_classes, normalizer=1.0)]
    
    storeData(data, f'all_hps_{outname}', root_dir)
