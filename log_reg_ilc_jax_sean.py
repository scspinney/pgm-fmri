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
from optax._src import transform, base
from jax import jit, grad, vmap
from jax.tree_util import tree_structure
import pickle
from collections import Counter

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
    ds = dataset.cache().repeat()
    # if is_training:
    #     ds = ds.shuffle(10 * batch_size, seed)
    ds = ds.batch(batch_size)

    return iter(ds.as_numpy_iterator())





# def load_dataset(
#     split: str,
#     *,
#     is_training: bool,
#     batch_size: int,
#     seed: int,
#     dataset
#     ) -> Generator[Batch, None, None]:
#     """Loads the dataset as a generator of batches."""
#     ds = dataset.cache().repeat()
#     if is_training:
#         ds = ds.shuffle(10 * batch_size, seed)
#     ds = ds.batch(batch_size)
#     return iter(ds.as_numpy_iterator())


### JAX Code
class ANDMaskState(optax.OptState):
  """Stateless.""" # Following optax code style

def and_mask(agreement_threshold: float) -> optax.GradientTransformation:
  def init_fn(_):
    # Required by optax
    return ANDMaskState()

  def update_fn(updates, opt_state, params=None):

    def and_mask(update):
      # Compute the masked gradients for a single parameter tensor
      mask = jnp.abs(jnp.mean(jnp.sign(update), 0)) >= agreement_threshold
      mask = mask.astype(jnp.float32)
      avg_update = jnp.mean(update, 0)
      mask_t = mask.sum() / mask.size
      update = mask * avg_update * (1. / (1e-10 + mask_t))
      return update

    del params # Following optax code style
    
    # Compute the masked gradients over all parameters

    # jax.tree_map maps a function (lambda function in this case) over a pytree to produce a new pytree.
    updates = jax.tree_map(lambda x: and_mask(x), updates)
    return updates, opt_state

  return base.GradientTransformation(init_fn, update_fn)


def sparse_logistic_regression(train=None, test=None, adam_lr=1e-3, agreement_threshold=0.0,
                               use_ilc=False, l1_coef=1e-5, l2_coef=1e-4,
                               epochs=10001, Verbose=False, n_classes=2, normalizer=255., training=True,seed=0):


    training_accs = []
    testing_accs = []

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
    def loss(params: hk.Params, batch, label) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        logits = net.apply(params, batch)
        labels = jax.nn.one_hot(label, n_classes)

        # Cross Entropy Loss
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]
        return softmax_xent 

        
    # Regularization loss (L1,L2).
    def regularization_loss(params: hk.Params) -> jnp.ndarray:
        """Compute the regularization loss of the network, applied after ILC."""

        # L1 Loss
        sum_in_layer = lambda p: jnp.sum(jnp.abs(p))
        sum_p_layers = [sum_in_layer(p) for p in jax.tree_leaves(params)]
        l1_loss = sum(sum_p_layers)

        # L2 Loss
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))

        return l2_coef * l2_loss + l1_coef * l1_loss
    

    @jax.jit
    def gradient_per_sample(params, batch, label):
        batch, label = jnp.expand_dims(batch,1), jnp.expand_dims(label,1)
        return vmap(grad(loss), in_axes=(None, 0, 0))(params, batch, label)

    gradient = jax.jit(grad(loss))
    gradient_reg = jax.jit(grad(regularization_loss))

    # Evaluation metric (classification accuracy).
    @jax.jit
    def accuracy(params: hk.Params, batch, label) -> jnp.ndarray:
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == label)
   
    @jax.jit
    def update(
        params: hk.Params,
        opt_state: OptState,
        batch, label, agreement
        ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        # grads = jax.grad(loss)(params, batch, label)
        # grads_masked = (gradient_per_sample if use_ilc else gradient)(params, batch, label) # (gradient_per_sample)(params, batch, label)
        # sum_grad_masked_regularized = jax.tree_multimap(lambda x,y:x+y,grads_masked,gradient_reg(params))
        # grads = sum_grad_masked_regularized
        # updates, opt_state = opt.update(grads, opt_state)
        # new_params = optax.apply_updates(params, updates)

        grads_samples = gradient_per_sample(params, batch, label)
        ANDmask = and_mask(agreement)

        masked_grads,_ = ANDmask.update(grads_samples, opt_state)
        reg_grads = gradient_reg(params)

        sum_grad_masked_regularized = jax.tree_multimap(lambda x,y:x+y,masked_grads,reg_grads)
 
        updates,_ = opt.update(sum_grad_masked_regularized, opt_state)

        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state


    # We maintain avg_params, the exponential moving average of the "live" params.
    # avg_params is used only for evaluation.
    # For more, see: https://doi.org/10.1137/0330046
    @jax.jit
    def ema_update(
        avg_params: hk.Params,
        new_params: hk.Params,
        epsilon: float = 0.001,
    ) -> hk.Params:
        return jax.tree_multimap(lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2,
                                avg_params, new_params)


    if training is False:
        return net, accuracy
    else:

        if use_ilc:

            use_ilc = False

            opt = optax.chain(optax.adam(adam_lr)
                # ,optax.scale_by_adam()
                )
            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(seed), next(train)[0])
            opt_state = opt.init(params)

            # opt = optax.chain(and_mask(agreement_threshold) if use_ilc else optax.identity(),optax.adam(adam_lr))
            # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used

            # Train/eval loop. WITHOUT ILC
            print("Begin training with ILC")
            for step in range(np.int(.5*epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    X, y = next(train)
                    train_accuracy = accuracy(avg_params, X, y)
                    train_accuracy = jax.device_get(train_accuracy)
                    X, y = next(test)
                    test_accuracy = accuracy(avg_params, X, y)
                    test_accuracy = jax.device_get(test_accuracy)
                    training_accs.append(train_accuracy)
                    testing_accs.append(test_accuracy)
                    if Verbose:
                        print(f"[ILC Off, Step {step}] Train accuracy/Test accuracy: "
                                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                X, y = next(train)
                params, opt_state = update(params, opt_state, X, y, 0.)
                avg_params = ema_update(avg_params, params)

            
            use_ilc = True

            # Train/eval loop. WITH ILC
            for step in range(np.int(.5*epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    X, y = next(train)
                    train_accuracy = accuracy(avg_params, X, y)
                    train_accuracy = jax.device_get(train_accuracy)
                    X, y = next(test)
                    test_accuracy = accuracy(avg_params, X, y)
                    test_accuracy = jax.device_get(test_accuracy)
                    training_accs.append(train_accuracy)
                    testing_accs.append(test_accuracy)
                    if Verbose:
                        print(f"[ILC On, Step {step}] Train accuracy/Test accuracy: "
                                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                X, y = next(train)
                params, opt_state = update(params, opt_state, X, y, agreement_threshold)
                avg_params = ema_update(avg_params, params)

            return params, training_accs, testing_accs

        else:

             # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used
            opt = optax.chain(optax.adam(adam_lr))

            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(seed), next(train)[0])
            opt_state = opt.init(params)

            use_ilc=False

            # Train/eval loop. 
            print("Begin training without ILC")
            for step in range(np.int(epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    X,y =  next(train)
                    train_accuracy = accuracy(avg_params, X, y)
                    train_accuracy = jax.device_get(train_accuracy)
                    X, y = next(test)
                    test_accuracy = accuracy(avg_params, X, y)
                    test_accuracy = jax.device_get(test_accuracy)
                    training_accs.append(train_accuracy)
                    testing_accs.append(test_accuracy)
                    if Verbose:
                        print(f"[ADAM, Step {step}] Train accuracy/Test accuracy: "
                                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")
                        
                # Do SGD on a batch of training examples.
                X, y = next(train)
                params, opt_state = update(params, opt_state, X, y, 0.)
                avg_params = ema_update(avg_params, params)
            
            return params, training_accs, testing_accs



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--use_ilc', default=False, action='store_true')
    parser.add_argument("--outname", type=str)

    args = parser.parse_args()

    if args.path == None:
        root_dir = "/Users/sean/Projects/pgm-fmri/data"
        batch_size = 2
        outname = "testing-01"
        use_ilc = False
        seed = 33
        n_classes = 2
        epochs = 100

    else:
        root_dir = args.path
        batch_size = args.batch_size
        outname = args.outname
        use_ilc = args.use_ilc
        seed = args.seed
        n_classes = args.n_classes
        epochs = args.epochs

    print("the path is ", root_dir)

    tf.config.experimental.set_visible_devices([], "GPU")


    at = [0.0, 0.5, 0.9]
    ll1 = [1e-1, 1e-2, 1e-3]
    ll2 = [1e-1, 1e-2, 1e-3]
    all = []
    n_envs = 1
    ds_train_envs = []
    print(f"Batch size: {batch_size}")

    train_years = [1,2]
    test_years = [3]

    train_ds = read_fmri_data(root_dir,train_years)
    train_ds = load_dataset("train", is_training=True, batch_size=batch_size, seed=seed, dataset=train_ds)

    test_ds = read_fmri_data(root_dir, test_years)
    test_ds = load_dataset("test", is_training=False, batch_size=batch_size, seed=seed, dataset=test_ds)

    round = 0
    for idx, thresh in enumerate(at):
        for l1 in ll1:
            for l2 in ll2:
                round += 1
                print('Round: ', round)

                envs_elastic_net_params = []
                hp = {}
                hp['thresh'] = thresh
                hp['l1'] = l1
                hp['l2'] = l2
                hp['params'] = []
                hp['training_accuracies'] = []
                hp['testing_accuracies'] = []
                for m in range(n_envs):
                    print('Parameters=[l1={}, l2={}, agreement={}], Environment={}'.format(l1,l2,thresh, m))
                    params, train_accs, test_accs = sparse_logistic_regression(train_ds, test_ds, adam_lr=1e-3, agreement_threshold=thresh,
                                            use_ilc=use_ilc, l1_coef=l1, l2_coef=l2,
                                            epochs=epochs, Verbose=True, n_classes=n_classes, normalizer=1, seed=seed)
                    envs_elastic_net_params.append(params)
                    hp['params'].append(params)
                    hp['training_accuracies'].append(train_accs)
                    hp['testing_accuracies'].append(test_accs)
                all.append(hp)

    storeData(all, f'all_hps_{outname}', root_dir)
