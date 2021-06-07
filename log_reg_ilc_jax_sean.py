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
    ds = dataset
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size).cache().repeat()
    return iter(ds.as_numpy_iterator())


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
            params = avg_params = net.init(jax.random.PRNGKey(42), next(train)['X'])
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument('--use_ilc', default=False, action='store_true')
    parser.add_argument("--outname", type=str)
    args = parser.parse_args()

    root_dir = args.path
    batch_size = args.batch_size
    outname = args.outname
    use_ilc = args.use_ilc

    print("the path is ", root_dir)

    tf.config.experimental.set_visible_devices([], "GPU")

    X_1 = list(tf.data.Dataset.list_files(root_dir + "/V1/X/*.npy").as_numpy_iterator())
    X_2 = list(tf.data.Dataset.list_files(root_dir + "/V2/X/*.npy").as_numpy_iterator())
    X_3 = list(tf.data.Dataset.list_files(root_dir + "/V3/X/*.npy").as_numpy_iterator())

    y_1 = list(tf.data.Dataset.list_files(root_dir + "/V1/y/*.npy").as_numpy_iterator())
    y_2 = list(tf.data.Dataset.list_files(root_dir + "/V2/y/*.npy").as_numpy_iterator())
    y_3 = list(tf.data.Dataset.list_files(root_dir + "/V3/y/*.npy").as_numpy_iterator())


    V1 = {'X': {}, 'y':{}}
    V2 = {'X': {}, 'y':{}}
    V3 = {'X': {}, 'y':{}}


    for i in range(len(X_1)):          
        try:
            index = int(X_1[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
            V1['X'][index-1] = np.load(X_1[i], mmap_mode='r')
        except:
            print(f"An exception occurred at index {i} of V1: {X_1[i]}")
            continue

        try:
            index = int(y_1[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])        
            V1['y'][index-1] = np.load(y_1[i], mmap_mode='r')
        except:
            print(f"An exception occurred at index {i} of V1: {y_1[i]}")
            continue


    for i in range(len(X_2)):     
        try:
            index = int(X_2[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
            V2['X'][index-1] = np.load(X_2[i], mmap_mode='r')
        except:
             print(f"An exception occurred at index {i} of V2: {X_2[i]}")
             continue
        try:
            index = int(y_2[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
            V2['y'][index-1] = np.load(y_2[i], mmap_mode='r')
        except:
            print(f"An exception occurred at index {i} of V2: {y_2[i]}")
            continue


    for i in range(len(X_3)):
        try:
            index = int(X_3[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
            V3['X'][index-1] = np.load(X_3[i], mmap_mode='r')
        except:
            print(f"An exception occurred at index {i} if V3: {X_3[i]}")
            continue

        try:
            index = int(y_3[i].decode("utf-8").split('/')[-1].split('.')[0].split('_')[-1])
            V3['y'][index-1] = np.load(y_3[i], mmap_mode='r')
        except:
            print(f"An exception occurred at index {i} if V3: {y_3[i]}")
            continue


    
    V1_X = []
    V2_X = []
    V3_X = []

    V1_y = []
    V2_y = []
    V3_y = []

    for key in sorted(V1['X']):
        try:
            V1_X.append(V1['X'][key])
            V1_y.append(V1['y'][key])
        except:
            print(f"An exception occurred for key {key} in V1.")
            print(V1['X'][key].shape)
            print(V1['y'][key].shape)
            continue

    for key in sorted(V2['X']):
        try:
            V2_X.append(V2['X'][key])
            V2_y.append(V2['y'][key]) 
        except:
            print(f"An exception occurred for key {key} in V2.")
            print(V2['X'][key].shape)
            print(V2['y'][key].shape)
            continue

    for key in sorted(V3['X']):
        try:
            V3_X.append(V3['X'][key])
            V3_y.append(V3['y'][key]) 
        except:
            print(f"An exception occurred for key {key} in V3.")
            print(V3['X'][key].shape)
            print(V3['y'][key].shape)
            continue

    V1_X = np.concatenate(V1_X, axis=0)
    V2_X = np.concatenate(V2_X, axis=0)
    V3_X = np.concatenate(V3_X, axis=0)

    V1_y = np.concatenate(V1_y, axis=0)
    V2_y = np.concatenate(V2_y, axis=0)
    V3_y = np.concatenate(V3_y, axis=0)

    # print(V1_X.shape)
    # print(V1_y.shape)

    # print(V1_X)
    # print(V1_y)

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

    ds_train_envs = []
    for d in datasets:
        ds = load_dataset("train", is_training=True, batch_size=batch_size, dataset=d)
        ds_train_envs.append(ds)
