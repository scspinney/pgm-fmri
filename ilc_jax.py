

"""### Model Definition

### Model definition in JAX
"""

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

  return optax.GradientTransformation(init_fn, update_fn)

from typing import Any, Generator, Mapping, Tuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

# Parameters
max_epochs = 5
weight_decay = 1e-5
n_batch = 32
n_workers = 6
learning_rate = 0.001
momentum = 0.9
l1_loss_coeff = 0.0005

fc_dim = mixed_dataset_train[0][0].shape[0]


# Generators
training_set = mixed_dataset_train
training_generator = DataLoader(training_set, batch_size=n_batch,shuffle=True, num_workers=n_workers, drop_last=True)

validation_set = mixed_dataset_test
validation_generator = DataLoader(validation_set, batch_size=n_batch, shuffle=True, num_workers=n_workers, drop_last=True)


def net_fn(batch) -> jnp.ndarray:
  """ """
  x = batch.astype(jnp.float32)
  mdl = hk.Sequential([
      hk.Linear(fc_dim), jax.nn.relu,
      hk.Linear(fc_dim), jax.nn.relu,
      hk.Linear(2),
  ])
  return mdl(x)


# Training loss (cross-entropy).
def loss(params: hk.Params, batch, labels, xent_weight, l1_coeff=0, l2_coeff=0) -> jnp.ndarray:
    """Compute the loss of the network, including L1, L2."""
    logits = net.apply(params, batch)
    labels = jax.nn.one_hot(label, 2)

    # Note that in our problem, regularization should be after the AND-mask
    sum_in_layer = lambda p: jnp.sum(p)
    sum_p_layers = [sum_in_layer(p) for p in jax.tree_leaves(params)]
    l1_loss = sum(sum_p_layers)
    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + l2_coeff * l2_loss + l1_coeff * l1_loss


# Evaluation metric (classification accuracy).
@jax.jit
def accuracy(params: hk.Params, batch, labels) -> jnp.ndarray:
    predictions = net.apply(params, batch)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == labels)

@jax.jit
def update(params: hk.Params, opt_state: OptState, batch, labels, xent_weight, l1_coeff=0, l2_coeff=0) -> Tuple[hk.Params, OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch, labels, xent_weight, l1_coeff, l2_coeff)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# if needed to take the gradient w.r.t. more than one variable: jax.grad(loss, (0,1))(params, batch, label, weights, 0.0001,0.001)

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





# Make the network and optimiser.
net = hk.without_apply_rng(hk.transform(net_fn))

############ ILC ############
steps_per_epoch = mixed_dataset_train.__len__()
agreement_threshold = 0.3
schedule_fn = optax.piecewise_constant_schedule(
      -learning_rate, # Note the minus sign!
      {40*steps_per_epoch: 0.1, 60*steps_per_epoch: 0.1})
optimizer = optax.chain(
    and_mask(agreement_threshold),
    optax.adam(1e-3),
    optax.scale_by_adam(),
    # optax.additive_weight_decay(),
    # optax.scale_by_schedule(schedule_fn)
    )
opt = optimizer
# If not ILC, use below optimizer
# opt = optax.adam(1e-3)

# Initialize network and optimiser; note we draw an input to get shapes.
params = avg_params = net.init(jax.random.PRNGKey(42), batch[0])
opt_state = opt.init(params)


# Train/eval loop.
for step in range(21):
    if step % 1 == 0:
        # Periodically evaluate classification accuracy on train & test sets.
        [batch, label] = next(iter(training_generator))
        batch, label = batch.detach().numpy(), label.detach().numpy()
        train_accuracy = accuracy(avg_params, batch, label)
        [batch, label] = next(iter(validation_generator))
        batch, label = batch.detach().numpy(), label.detach().numpy()
        test_accuracy = accuracy(avg_params, batch, label)
        train_accuracy, test_accuracy = jax.device_get(
            (train_accuracy, test_accuracy))
        print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    # Do SGD on a batch of training examples.
    [batch, label] = next(iter(training_generator))
    batch, label = batch.detach().numpy(), label.detach().numpy()
    params, opt_state = update(params, opt_state, batch, label, weights, 0.0001,0.001)
    avg_params = ema_update(avg_params, params)
