import os
import numpy as np
import tensorflow as tf
import argparse
from typing import Any, Generator, Mapping
import numpy as np
np.set_printoptions(precision=5, suppress=True)


OptState = Any
Batch = Mapping[str, np.ndarray]


def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
    ds = ds.shuffle(10000)
    return ds


def process_fmri_data(root_dir, years, labels, outpath, test=False):
    def process_year(y):
        if test:
            X_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}_test/X*.npy").as_numpy_iterator())
            y_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}_test/y*.npy").as_numpy_iterator())
        else:
            X_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}/X*.npy").as_numpy_iterator())
            y_1 = list(tf.data.Dataset.list_files(root_dir + f"/V{y}/y*.npy").as_numpy_iterator())

        V1 = {'X': {}, 'y': {}}

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
        min_class = min(labels)
        X_pos = X[y == labels[0]]
        X_neg = X[y == labels[1]]
        y_pos = y[y == labels[0]] % min_class
        y_neg = y[y == labels[1]] % min_class


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
        # datasets.append(tf.data.Dataset.from_tensor_slices({'X': X, 'y': Y}))

    pos_ds = pos_ds_l[0].concatenate(pos_ds_l[1]) if len(pos_ds_l) > 1 else pos_ds_l[0]
    neg_ds = neg_ds_l[0].concatenate(neg_ds_l[1]) if len(neg_ds_l) > 1 else neg_ds_l[0]
    resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

    tf.data.experimental.save(
        resampled_ds, outpath, compression=None, shard_func=None
    )
    print(f"Successfully saved dataset to {outpath}")



def load_binary_dataset(
        split: str,
        *,
        is_training: bool,
        batch_size: int,
        seed: int,
        path: str
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tf.data.experimental.load(path)
    ds = ds.cache().repeat()
    # if is_training:
    #     ds = ds.shuffle(10 * batch_size, seed)
    ds = ds.batch(batch_size)

    return iter(ds.as_numpy_iterator())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--test_load', default=False, action='store_true')
    parser.add_argument('--use_subset', default=False, action='store_true')

    args = parser.parse_args()

    if args.path == None:
        root_dir = "/Users/sean/Projects/pgm-fmri/data"
        batch_size = 2
        use_ilc = False
        seed = 33
        n_classes = 2
        epochs = 100
        test_load = True
        use_subset = True

    else:
        root_dir = args.path
        batch_size = args.batch_size
        use_ilc = args.use_ilc
        seed = args.seed
        n_classes = args.n_classes
        epochs = args.epochs
        test_load = args.test_load
        use_subset = args.use_subset

    print("the path is ", root_dir)

    train_years = [1, 2]
    test_years = [3]
    labels = [4, 5]
    print(f"Processing training on labels: {labels}")
    out_train = os.path.join(root_dir,f"training_{'test' if use_subset else 'complete'}_ds")
    train_ds = process_fmri_data(root_dir, train_years, labels, out_train,test=use_subset)
    print(f"Processing test...")
    out_test = os.path.join(root_dir,f"testing_{'test' if use_subset else 'complete'}_ds")
    test_ds = process_fmri_data(root_dir, test_years, labels, out_test,test=use_subset)

    if test_load:
        train_ds = load_binary_dataset("train", is_training=True, batch_size=batch_size, seed=seed, path=out_train)
        test_ds = load_binary_dataset("test", is_training=False, batch_size=batch_size, seed=seed, path=out_test)

        # test train
        for train,test in zip(train_ds,train_ds):
            print(f"TRAIN: X shape: {train[0].shape}, y label/shape:{train[1]}, {train[1].shape}")
            print(f"TEST: X shape: {test[0].shape}, y label/shape:{test[1]}, {test[1].shape}")
            break
