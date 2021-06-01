import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
np.set_printoptions(precision=5, suppress=True)
import tensorflow as tf
import tensorflow_datasets as tfds

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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from matplotlib.pyplot import figure
import pickle

def storeData(object, file_name, root_dir):
    with open(root_dir+file_name, 'wb') as f:
        pickle.dump(object, f)					 
        f.close() 

def loadData(file_name, root_dir): 
    with open(root_dir+file_name, 'rb') as f:
        db = pickle.load(f) 
        f.close()
        return db

"""### The last feature is the only robust one (it can be found in any environment), but it produces a weaker signal, and has a higher cost using weight decay.

"""

# ADD_SMALL_NOISE = False

# # Create the dataset.
# x = 3 * torch.cat((torch.eye(4), 0.1 * torch.ones(4).view(-1,1)), dim=1)
# y = torch.tensor([1., 1., 1., 1.]).float()

# if ADD_SMALL_NOISE:
#     dist = torch.distributions.Uniform(-0.001, 0.001)
    
#     # This adds noise on all non-robust feature
#     x[:, :-1] += dist.sample(x[:, :-1].shape)
    
    # This adds noise on every feature
    #     x += dist.sample(x.shape)

# print('x:', x.numpy(), sep="\n")
# print('y: ', y.view(-1, 1).numpy(), sep="\n")

# Make datasets.
# partitions = {'train':[0,1], 'test': [2,3]}
# print(x.numpy()[partitions['train']])
# dataset_train = {'x':x.numpy()[partitions['train']], 'y':y.numpy()[partitions['train']]}
# dataset_test = {'x':x.numpy()[partitions['test']], 'y':y.numpy()[partitions['test']]}

OptState = Any
Batch = Mapping[str, np.ndarray]

def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
    dataset
    ) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = dataset.cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


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

  return transform.GradientTransformation(init_fn, update_fn)


def sparse_logistic_regression(train=None, test=None, adam_lr=1e-3, agreement_threshold=0.0,
                               use_ilc=False, l1_coef=1e-5, l2_coef=1e-4,
                               epochs=10001, Verbose=False, n_classes=2, normalizer=255., training=True):


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
            params = avg_params = net.init(jax.random.PRNGKey(42), next(train)['X'])
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
                    Batch = next(train)
                    train_accuracy = accuracy(avg_params, Batch['X'], Batch['y'])
                    train_accuracy = jax.device_get(train_accuracy)
                    Batch = next(test)
                    test_accuracy = accuracy(avg_params, Batch['X'], Batch['y'])
                    test_accuracy = jax.device_get(test_accuracy)
                    training_accs.append(train_accuracy)
                    testing_accs.append(test_accuracy)
                    if Verbose:
                        print(f"[ILC Off, Step {step}] Train accuracy/Test accuracy: "
                                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                Batch = next(train)
                params, opt_state = update(params, opt_state, Batch['X'], Batch['y'], 0.)
                avg_params = ema_update(avg_params, params)

            
            use_ilc = True

            # Train/eval loop. WITH ILC
            for step in range(np.int(.5*epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    Batch = next(train)
                    train_accuracy = accuracy(avg_params, Batch['X'], Batch['y'])
                    train_accuracy = jax.device_get(train_accuracy)
                    Batch = next(test)
                    test_accuracy = accuracy(avg_params, Batch['X'], Batch['y'])
                    test_accuracy = jax.device_get(test_accuracy)
                    training_accs.append(train_accuracy)
                    testing_accs.append(test_accuracy)
                    if Verbose:
                        print(f"[ILC On, Step {step}] Train accuracy/Test accuracy: "
                                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                Batch = next(train)
                params, opt_state = update(params, opt_state, Batch['X'], Batch['y'], agreement_threshold)
                avg_params = ema_update(avg_params, params)

            return params, training_accs, testing_accs

        else:

             # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used
            opt = optax.chain(optax.adam(adam_lr))

            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(42), next(train)[0])
            opt_state = opt.init(params)

            use_ilc=False

            # Train/eval loop. 
            print("Begin training without ILC")
            for step in range(np.int(epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    Batch = next(train)
                    train_accuracy = accuracy(avg_params, Batch['X'], Batch['y'])
                    train_accuracy = jax.device_get(train_accuracy)
                    Batch = next(test)
                    test_accuracy = accuracy(avg_params, Batch['X'], Batch['y'])
                    test_accuracy = jax.device_get(test_accuracy)
                    training_accs.append(train_accuracy)
                    testing_accs.append(test_accuracy)
                    if Verbose:
                        print(f"[ADAM, Step {step}] Train accuracy/Test accuracy: "
                                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")
                        
                # Do SGD on a batch of training examples.
                Batch = next(train)
                params, opt_state = update(params, opt_state, Batch['X'], Batch['y'], 0.)
                avg_params = ema_update(avg_params, params)
            
            return params, training_accs, testing_accs


# Training with Adam, no reg, no ilc
# p1 = sparse_logistic_regression(train=dataset_train, test=dataset_test, adam_lr=0.4, agreement_threshold=0.0,
#                                use_ilc=False, l1_coef=0., l2_coef=0.,
#                                epochs=10001, Verbose=True, training=True)

# # Training with Adam, with reg, with ilc
# p2 = sparse_logistic_regression(train=dataset_train, test=dataset_test, adam_lr=0.4, agreement_threshold=1.,
#                                use_ilc=True, l1_coef=1e-4, l2_coef=1e-4,
#                                epochs=10001, Verbose=True, training=True)

# Compare the results with GBs derived weights
# print(p1) # Linear Regression when SGD/Adam is used
# print(p2) # Linear Regression when SGD/Adam is used with Reg and ILC

if __name__ == "__main__":

    root_dir = ''

    X_1 = list(tf.data.Dataset.list_files("V1/X/*.npy").as_numpy_iterator())
    X_2 = list(tf.data.Dataset.list_files("V2/X/*.npy").as_numpy_iterator())
    X_3 = list(tf.data.Dataset.list_files("V3/X/*.npy").as_numpy_iterator())

    y_1 = list(tf.data.Dataset.list_files("V1/y/*.npy").as_numpy_iterator())
    y_2 = list(tf.data.Dataset.list_files("V2/y/*.npy").as_numpy_iterator())
    y_3 = list(tf.data.Dataset.list_files("V3/y/*.npy").as_numpy_iterator())

    V1_X = np.zeros((len(X_1), 300, 53, 63, 52))
    V2_X = np.zeros((len(X_2), 300, 53, 63, 52))
    V3_X = np.zeros((len(X_3), 300, 53, 63, 52))

    V1_y = np.zeros((len(y_1), 300))
    V2_y = np.zeros((len(y_2), 300))
    V3_y = np.zeros((len(y_3), 300))

    for i in range(len(X_1)):
        index = int(X_1[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
        V1_X[index-1] = np.load(X_1[i])
        index = int(y_1[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
        V1_y[index-1] = np.load(y_1[i])

    for i in range(len(X_2)):
        index = int(X_2[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
        V2_X[index-1] = np.load(X_2[i])
        index = int(y_2[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
        V2_y[index-1] = np.load(y_2[i])

    for i in range(len(X_3)):
        index = int(X_3[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
        V3_X[index-1] = np.load(X_3[i])
        index = int(y_3[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
        V3_y[index-1] = np.load(y_3[i])

    # remove class 0 or go_scan
    V1_X = V1_X[V1_y != 0]
    V2_X = V2_X[V2_y != 0]
    V3_X = V3_X[V3_y != 0]

    V1_y = V1_y[V1_y != 0]
    V2_y = V2_y[V2_y != 0]
    V3_y = V3_y[V3_y != 0]

    V1_X = V1_X.reshape(-1,53, 63, 52)
    V2_X = V2_X.reshape(-1,53, 63, 52)
    V3_X = V3_X.reshape(-1,53, 63, 52)

    V1_y = V1_y.reshape(-1)
    V2_y = V2_y.reshape(-1)
    V3_y = V3_y.reshape(-1)

    # normalize data
    V1_X = (V1_X - V1_X.mean()) / V1_X.std()
    V2_X = (V2_X - V2_X.mean()) / V2_X.std()
    V3_X = (V3_X - V3_X.mean()) / V3_X.std()

    datasets = [tf.data.Dataset.from_tensor_slices({'X': V1_X, 'y': V1_y}),
                tf.data.Dataset.from_tensor_slices({'X': V2_X, 'y': V2_y}),
                tf.data.Dataset.from_tensor_slices({'X': V3_X, 'y': V3_y})]

    at = [0.0, 0.5, 0.9]
    ll1 = [1e-1, 1e-2, 1e-3]
    ll2 = [1e-1, 1e-2, 1e-3]
    all = []
    n_envs = 2
    ds_train_envs = []
    batch_size = 100
    for m in range(n_envs):
        ds = load_dataset("train", is_training=True, batch_size=batch_size, dataset=datasets[m])
        ds_train_envs.append(ds)

    test_ds = load_dataset("test", is_training=False, batch_size=batch_size, dataset=datasets[-1])

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
                    params, train_accs, test_accs = sparse_logistic_regression(ds_train_envs[m], test_ds, adam_lr=1e-3, agreement_threshold=thresh,
                                            use_ilc=True, l1_coef=l1, l2_coef=l2,
                                            epochs=10001, Verbose=True, n_classes=5, normalizer=1)
                    envs_elastic_net_params.append(params)
                    hp['params'].append(params)
                    hp['training_accuracies'].append(train_accs)
                    hp['testing_accuracies'].append(test_accs)
                all.append(hp)

    storeData(all, 'all_hps_3', root_dir)