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
!pip install -q git+git://github.com/deepmind/optax.git
!pip install -q git+https://github.com/deepmind/dm-haiku
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

"""### The last feature is the only robust one (it can be found in any environment), but it produces a weaker signal, and has a higher cost using weight decay.

"""

ADD_SMALL_NOISE = False

# Create the dataset.
x = 3 * torch.cat((torch.eye(4), 0.1 * torch.ones(4).view(-1,1)), dim=1)
y = torch.tensor([1., 1., 1., 1.]).float()

if ADD_SMALL_NOISE:
    dist = torch.distributions.Uniform(-0.001, 0.001)
    
    # This adds noise on all non-robust feature
    x[:, :-1] += dist.sample(x[:, :-1].shape)
    
    # This adds noise on every feature
    #     x += dist.sample(x.shape)

print('x:', x.numpy(), sep="\n")
print('y: ', y.view(-1, 1).numpy(), sep="\n")

# Make datasets.
partitions = {'train':[0,1], 'test': [2,3]}
print(x.numpy()[partitions['train']])
dataset_train = {'x':x.numpy()[partitions['train']], 'y':y.numpy()[partitions['train']]}
dataset_test = {'x':x.numpy()[partitions['test']], 'y':y.numpy()[partitions['test']]}

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

OptState = Any
Batch = Mapping[str, np.ndarray]

def linear_regression(dataset_train=None, dataset_test=None, adam_lr=0.3, agreement_threshold=0.0,
                               use_ilc=False, l1_coef=1e-4, l2_coef=1e-4,
                               epochs=1001, Verbose=False, training=True):
    training_loss = []
    testing_loss = []
    def net_fn(batch) -> jnp.ndarray:
        x = jnp.array(batch, jnp.float32)
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(1, with_bias=False)
        ])
        return mlp(x)


    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))
        
    # Training loss (cross-entropy).
    def loss(params: hk.Params, batch, label) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        logits = net.apply(params, batch)
        
        msl = 0.5 * jnp.sum(jnp.power(logits - label,2)) / batch.shape[0]

        return msl 

        
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

    # For regression there's no accuracy, if you changed to classification, uncomment this:

    # Evaluation metric (classification accuracy).
    # @jax.jit
    # def accuracy(params: hk.Params, batch, label) -> jnp.ndarray:
    #     predictions = net.apply(params, batch)
    #     return jnp.mean(jnp.argmax(predictions, axis=-1) == label)

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: OptState,
        batch, label, agreement
        ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        # grads_masked = (gradient_per_sample if use_ilc else gradient)(params, batch, label) # (gradient_per_sample)(params, batch, label)
        # sum_grad_masked_regularized = jax.tree_multimap(lambda x,y:x+y,grads_masked,gradient_reg(params))
        # grads = sum_grad_masked_regularized
        # updates, opt_state = opt.update(grads, opt_state)
        # new_params = optax.apply_updates(params, updates)

        # grads = gradient(params, batch, label)
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
        epsilon: float = 0.01,
    ) -> hk.Params:
        return jax.tree_multimap(lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2,
                                avg_params, new_params)
    
    if training is False:
        return net, accuracy
    else:
        if(use_ilc):

            use_ilc = False

            # opt = optax.chain(and_mask(agreement_threshold) if use_ilc else optax.identity(),optax.adam(adam_lr))

            # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used
            opt = optax.chain(optax.adam(adam_lr)) # Or Adam could be used

            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(42), dataset_train['x'][0])
            opt_state = opt.init(params)


            # Train/eval loop. WITHOUT ILC
            for step in range(np.int(epochs/2)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    batch, y = dataset_train['x'], dataset_train['y']
                    train_loss = loss(avg_params, batch, y)
                    batch, y = dataset_test['x'], dataset_test['y']
                    test_loss = loss(avg_params, batch, y)
                    train_loss, test_loss = jax.device_get(
                        (train_loss, test_loss))
                    training_loss.append(train_loss)
                    testing_loss.append(test_loss)
                    if Verbose:
                        print(f"[ILC Off, Step {step}] Test loss: "
                            f"{test_loss:.3f}") # f"{test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                batch, y = dataset_train['x'], dataset_train['y']
                params, opt_state = update(params, opt_state, batch, y, 0.)
                avg_params = ema_update(avg_params, params)
            

            use_ilc = True

            # opt = optax.chain(optax.adam(adam_lr))

            # Initialize network and optimiser; note we draw an input to get shapes.
            opt_state = opt.init(params)
            
            # Train/eval loop. WITH ILC
            for step in range(np.int(epochs/2)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    batch, y = dataset_train['x'], dataset_train['y']
                    train_loss = loss(avg_params, batch, y)
                    batch, y = dataset_test['x'], dataset_test['y']
                    test_loss = loss(avg_params, batch, y)
                    train_loss, test_loss = jax.device_get(
                        (train_loss, test_loss))
                    training_loss.append(train_loss)
                    testing_loss.append(test_loss)
                    if Verbose:
                        print(f"[ILC On, Step {step}] Test loss: "
                            f"{test_loss:.3f}") # f"{test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                batch, y = dataset_train['x'], dataset_train['y']
                params, opt_state = update(params, opt_state, batch, y, agreement_threshold)
                avg_params = ema_update(avg_params, params)

            return params, training_loss, testing_loss

        else:
            # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used
            opt = optax.chain(optax.adam(adam_lr))

            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(42), dataset_train['x'][0])
            opt_state = opt.init(params)


            # Train/eval loop.
            for step in range(epochs):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    batch, y = dataset_train['x'], dataset_train['y']
                    train_loss = loss(avg_params, batch, y)
                    batch, y = dataset_test['x'], dataset_test['y']
                    test_loss = loss(avg_params, batch, y)
                    train_loss, test_loss = jax.device_get(
                        (train_loss, test_loss))
                    training_loss.append(train_loss)
                    testing_loss.append(test_loss)
                    if Verbose:
                        print(f"[ADAM, Step {step}] Test loss: "
                            f"{test_loss:.3f}") # f"{test_accuracy:.3f}.")

                # Do SGD on a batch of training examples.
                batch, y = dataset_train['x'], dataset_train['y']
                params, opt_state = update(params, opt_state, batch, y, 0.)
                avg_params = ema_update(avg_params, params)
            
            return params, training_loss, testing_loss

# Training with Adam, no reg, no ilc
p1 = linear_regression(dataset_train=dataset_train, dataset_test=dataset_test, adam_lr=0.4, agreement_threshold=0.0,
                               use_ilc=False, l1_coef=0., l2_coef=0.,
                               epochs=10001, Verbose=True, training=True)

# Training with Adam, with reg, with ilc
p2 = linear_regression(dataset_train=dataset_train, dataset_test=dataset_test, adam_lr=0.4, agreement_threshold=1.,
                               use_ilc=True, l1_coef=1e-4, l2_coef=1e-4,
                               epochs=10001, Verbose=True, training=True)

# Compare the results with GBs derived weights
print(p1) # Linear Regression when SGD/Adam is used
print(p2) # Linear Regression when SGD/Adam is used with Reg and ILC

############################## Linear Regression with generators as input ##############################

OptState = Any
Batch = Mapping[str, np.ndarray]


def linear_regression(train=None, test=None, adam_lr=0.3, agreement_threshold=0.0,
                               use_ilc=False, l1_coef=1e-4, l2_coef=1e-4,
                               epochs=1001, Verbose=False, training=True, normalizer=255.):
  


    training_loss = []
    testing_loss = []

    def net_fn(batch) -> jnp.ndarray:
        x = jnp.array(batch, jnp.float32) / normalizer
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(1, with_bias=False)
        ])
        return mlp(x)

    

    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))
        
    # Training loss (cross-entropy).
    def loss(params: hk.Params, batch, label) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        logits = net.apply(params, batch)
        
        msl = 0.5 * jnp.sum(jnp.power(logits - label,2)) / batch.shape[0]

        return msl 

        
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

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: OptState,
        batch, label, agreement
        ) -> Tuple[hk.Params, OptState]:
        """Learning rule (stochastic gradient descent)."""
        # grads_masked = (gradient_per_sample if use_ilc else gradient)(params, batch, label) # (gradient_per_sample)(params, batch, label)
        # sum_grad_masked_regularized = jax.tree_multimap(lambda x,y:x+y,grads_masked,gradient_reg(params))
        # grads = sum_grad_masked_regularized
        # updates, opt_state = opt.update(grads, opt_state)
        # new_params = optax.apply_updates(params, updates)

        # grads = gradient(params, batch, label)
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
        epsilon: float = 0.01,
    ) -> hk.Params:
        return jax.tree_multimap(lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2,
                                avg_params, new_params)
    
    if training is False:
        return net
    else:
        if(use_ilc):

            use_ilc = False

            # opt = optax.chain(and_mask(agreement_threshold) if use_ilc else optax.identity(),optax.adam(adam_lr))

            # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used
            opt = optax.chain(optax.adam(adam_lr)) # Or Adam could be used

            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(42), next(train)[0])
            opt_state = opt.init(params)

            # Train/eval loop. WITHOUT ILC
            print("Begin training with ILC")
            for step in range(np.int(.5*epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    Batch = next(train)
                    train_loss = loss(avg_params, Batch[0], Batch[1])
                    train_loss = jax.device_get(train_loss)
                    Batch = next(test)
                    test_loss = loss(avg_params, Batch[0], Batch[1])
                    test_loss = jax.device_get(test_loss)
                    training_loss.append(train_accuracy)
                    testing_loss.append(test_accuracy)
                    if Verbose:
                        print(f"[ILC Off, Step {step}] Train loss/Test loss: "
                                f"{train_loss:.3f} / {test_loss:.3f}.")

                # Do SGD on a batch of training examples.
                Batch = next(train)
                params, opt_state = update(params, opt_state, Batch[0], Batch[1], 0.)
                avg_params = ema_update(avg_params, params)
            

            use_ilc = True

            
            # Train/eval loop. WITH ILC
            for step in range(np.int(.5*epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    Batch = next(train)
                    train_loss = loss(avg_params, Batch[0], Batch[1])
                    train_loss = jax.device_get(train_loss)
                    Batch = next(test)
                    test_loss = loss(avg_params, Batch[0], Batch[1])
                    test_loss = jax.device_get(test_loss)
                    training_loss.append(train_accuracy)
                    testing_loss.append(test_accuracy)
                    if Verbose:
                        print(f"[ILC On, Step {step}] Train loss/Test loss: "
                                f"{train_loss:.3f} / {test_loss:.3f}.")

                # Do SGD on a batch of training examples.
                Batch = next(train)
                params, opt_state = update(params, opt_state, Batch[0], Batch[1], agreement_threshold)
                avg_params = ema_update(avg_params, params)
          

            return params, training_loss, testing_loss

        else:
                
            # schedule_fn = optax.exponential_decay(adam_lr, # Note the minus sign!
            # 1,
            # 0.9)
            # opt = optax.chain(optax.sgd(adam_lr), optax.scale_by_schedule(schedule_fn)) # Or Adam could be used
            opt = optax.chain(optax.adam(adam_lr))

            # Initialize network and optimiser; note we draw an input to get shapes.
            params = avg_params = net.init(jax.random.PRNGKey(42), dataset_train['x'][0])
            opt_state = opt.init(params)

            # Train/eval loop. 
            print("Begin training without ILC")
            for step in range(np.int(epochs)):
                if step % np.int(epochs/10) == 0:
                    # Periodically evaluate classification accuracy on train & test sets.
                    Batch = next(train)
                    train_loss = loss(avg_params, Batch[0], Batch[1])
                    train_loss = jax.device_get(train_loss)
                    Batch = next(test)
                    test_loss = loss(avg_params, Batch[0], Batch[1])
                    test_loss = jax.device_get(test_loss)
                    training_loss.append(train_accuracy)
                    testing_loss.append(test_accuracy)
                    if Verbose:
                        print(f"[ADAM, Step {step}] Train loss/Test loss: "
                                f"{train_loss:.3f} / {test_loss:.3f}.")
                        
                # Do SGD on a batch of training examples.
                Batch = next(train)
                params, opt_state = update(params, opt_state, Batch[0], Batch[1], 0.)
                avg_params = ema_update(avg_params, params)

            
            return params, training_loss, testing_loss
