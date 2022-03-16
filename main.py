from typing import Callable, Tuple

from tqdm import tqdm

import matplotlib.pyplot as plt

import jax
from jax import random
import jax.numpy as np

from flax import linen as nn
from flax.training import train_state

import optax

import environment


class ValueFunction():
    pass


class EnvironmentModel(nn.Module):
    """"Generative model of N-dim input to N-dim output."""

    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=16)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_size)(x)
        return x


def hessian(fun):
    """Compute the hessian for an input function"""
    return jax.jit(jax.jacfwd(jax.jacrev(fun)))


def model_prediction(theta, model, inp):
    """Compute the model prediction for a given input."""
    return model.apply({'params': theta}, inp)


@jax.vmap(in_axes=(0,0,None))
def mse_loss(model_prediction, environment_sample):
    """Compute the MSE loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    return np.mean(np.square(err)) 


@jax.vmap(in_axes=(0,0,None))
def vagram_loss(model_prediction, environment_sample, value_function):
    """Compute the VAGRAM loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    _, grad = jax.value_and_grad(value_function)(environment_sample)
    return np.square(grad * err).sum()


@jax.vmap(in_axes=(0,0,None))
def quadratic_vagram_loss(model_prediction, environment_sample, value_function):
    """Compute the quadratic upper bounded 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample)
    l, Q = np.linalg.eigh(hes)
    basis_trans = np.dot(np.transpose(Q), err)
    return np.square(l * basis_trans).sum()


@jax.vmap(in_axes=(0,0,None))
def quadratic_vagram_loss_quartic(model_prediction, environment_sample, value_function):
    """Compute the original quartic 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample)
    l, Q = np.linalg.eigh(hes)
    basis_trans = np.dot(np.transpose(Q), err)
    return np.square(l * np.square(basis_trans)).sum()


@jax.vmap(in_axes=(0,0,None))
def eval_loss_vaml(model_prediction, environment_sample, value_function):
    """Compute the VAML loss for validation for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = value_function(environment_sample) - value_function(model_prediction)
    return np.square(err).sum()


@jax.vmap(in_axes=(0,0,None))
def eval_loss_mse(model_prediction, environment_sample, value_function):
    """Compute the MSE loss for validation for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    return np.square(err).sum()


def train_step(batch_x, batch_y, loss_function, model, optim_state, value_function):
    """Train step for a given batch of data. Wraps the gradient update step"""
    @jax.jit
    def loss(t):
        pred_y = model_prediction(t, model, batch_x)
        return loss_function(pred_y, batch_y, value_function)
    loss_value, grad = jax.value_and_grad(loss)(optim_state.params)
    optim_state = optim_state.apply_gradients(grads=grad)
    return loss_value, optim_state


def train(
    train_ds: Tuple[np.ndarray, np.ndarray], 
    val_ds: Tuple[np.ndarray, np.ndarray], 
    loss_function: Callable, 
    value_function: Callable):
    """
    Core training loop, expects a train and a validation dataset, as well as a loss function and a value function to compute VMAL and VaGram losses
    """

    # init PRNG
    key = random.PRNGKey(0)
    data_x = train_ds[0]
    data_y = train_ds[1]

    val_x = val_ds[0]
    val_y = val_ds[1]
  
    # hyperparameters
    inp_dim = data_x.shape[-1]
    lr = 3e-4
    epoch_num = 10
    n_samples = 100000
    batch_size = 32

    # init networks
    key, model_key = random.split(key)
    model = EnvironmentModel(output_size=data_y.shape[-1])
    key1, key2 = random.split(model_key)
    x = random.normal(key1, (inp_dim,)) # Dummy input
    params = model.init(key2, x)["params"] # Initialization call
    tx = optax.rmsprop(lr)
    optim_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # bookeeping for plotting
    all_loss_values = np.array([])
    all_loss_val = np.array([])
    all_vaml_val = np.array([])
    all_mse_val = np.array([])
  
    # train loop
    def train_epoch(optim_state, model, batch_size, epoch, rng):
        """Train for a single epoch."""
        rng, z_rng = random.split(rng)
        train_ds_size = len(data_x)
        steps_per_epoch = train_ds_size // batch_size
  
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        batch_metrics = []
        loss_values = []
        for perm in tqdm(perms):

            # get batch from permutations
            batch_x = data_x[perm]
            batch_y = data_y[perm]
  
            @jax.jit
            def f(batch_x, batch_y, state):
                return train_step(batch_x, batch_y, loss_function, model, state, value_function)
            loss_value, optim_state = f(batch_x, batch_y, optim_state)
            loss_values.append(loss_value)

        # validate on two reference losses: VAML and MSE
        val_model_prediction = model_prediction(val_x, model, val_x)
        val_loss = loss_function(val_model_prediction, val_y, value_function)
        vaml_val_loss = vaml_val_loss(val_model_prediction, val_y, value_function)
        mse_val_loss = mse_val_loss(val_model_prediction, val_y, value_function)

        return np.array(loss_values), val_loss, vaml_val_loss, mse_val_loss, optim_state

    for i in tqdm(range(epoch_num)):
        key, epoch_key = random.split(key)
        loss_values, loss_val, vaml_val, mse_val, optim_state = train_epoch(optim_state, model, batch_size, i, epoch_key)

        # saving the training loss
        all_loss_values = np.concatenate((all_loss_values, loss_values))

        # saving the validation losses
        all_loss_val.append(loss_val)
        all_vaml_val.append(vaml_val)
        all_mse_val.append(mse_val)

    plt.plot(all_loss_values)
    plt.show()

    plt.plot(all_vaml_val)
    plt.show()

    plt.plot(all_mse_val)
    plt.show()

    return all_loss_values


if __name__ == "__main__":
    env = environment.make_pendulum()
    obs, act = environment.get_samples(env)

    train_ratio = 0.9

    x = np.concatenate([obs, act], axis=-1)[:, :-1]
    x = x.reshape(-1, x.shape[-1])
    y = obs[:, 1:]
    y = y.reshape(-1, y.shape[-1])

    train_ds = (x[:int(len(x) * train_ratio)], y[:int(len(y) * train_ratio)])
    val_ds = (x[int(len(x) * train_ratio):], y[int(len(y) * train_ratio):])

    s = np.array([[1., 2., 3.], [2., 4., 9.]])
    s_target = np.array([[2., 0., 0.5], [0., 0., 0.]])
    
    value_function = lambda x: np.sum(np.sin(x))
    loss_function = lambda x, y, z: np.mean(quadratic_vagram_loss(x, y, z))

    all_loss_values = train(train_ds, val_ds, loss_function, value_function)