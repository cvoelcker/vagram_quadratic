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
  return jax.jit(jax.jacfwd(jax.jacrev(fun)))


def model_prediction(theta, model, inp):
    return model.apply({'params': theta}, inp)


def mse_loss(model_prediction, environment_sample):
    err = model_prediction - environment_sample
    return np.mean(np.square(err)) 


def vagram_loss(model_prediction, environment_sample, value_function):
    err = model_prediction - environment_sample
    _, grad = jax.value_and_grad(value_function)(environment_sample)
    return np.square(grad * err).sum()


def quadratic_vagram_loss(model_prediction, environment_sample, value_function):
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample)
    l, Q = np.linalg.eigh(hes)
    basis_trans = np.dot(np.transpose(Q), err)
    return np.square(l * basis_trans).sum()


def train_step(batch_x, batch_y, loss_function, model, state, value_function):
    @jax.jit
    def loss(t):
        pred_y = model_prediction(t, model, batch_x)
        return loss_function(pred_y, batch_y, value_function)
    value, grad = jax.value_and_grad(loss(state.params))
    state = state.apply_gradients(grads=grad)
    return value, state


def train(data_x, data_y, train_step):
    # init PRNG
    key = random.PRNGKey(0)
  
    # hyperparameters
    inp_dim = data_x.shape[-1]
    lr = 3e-4
    epoch_num = 10
    n_samples = 100000

    # init networks
    key, model_key = random.split(key)
    model = EnvironmentModel(output_size=data_y.shape[-1])
    key1, key2 = random.split(model_key)
    x = random.normal(key1, (inp_dim,)) # Dummy input
    params = model.init(key2, x)["params"] # Initialization call
    tx = optax.rmsprop(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  
    # train loop
    def train_epoch(state, model, value, train_ds, batch_size, epoch, rng):
        """Train for a single epoch."""
        rng, z_rng = random.split(rng)
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
  
        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        batch_metrics = []
        for perm in perms:
            batch_x = train_ds[0][perm]
            batch_y = train_ds[1][perm]
  
            #TODO: fix here later

        # eval
        z_eval = random.normal(z_rng, shape=(1000,16))
        eval_samples = model.apply({'params': state.params}, z_eval)
  
        return state, eval_samples


if __name__ == "__main__":
    env = environment.make_pendulum()
    obs, act = environment.get_samples(env)

    x = np.concatenate([obs, act], axis=-1)[:, :-1]
    x = x.reshape(-1, x.shape[-1])
    y = obs[:, 1:]
    y = y.reshape(-1, y.shape[-1])

    s = np.array([[1., 2., 3.], [2., 4., 9.]])
    s_target = np.array([[2., 0., 0.5], [0., 0., 0.]])

    print(jax.vmap(vagram_loss, in_axes=(0, 0, None))(s, s_target, lambda x: np.sum(np.sin(x))))

    # train()