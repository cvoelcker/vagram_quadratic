from flax import linen as nn
from jax.nn import initializers as init
from jax import numpy as jnp
from jax import random
import optax
from flax.training import train_state


class ModelNetwork(nn.Module):
    hidden_size: int
    state_dim: int
    action_dim: int
    ensemble_members: int

    @nn.compact
    def __call__(self, x):
        results = []
        for _ in range(self.ensemble_members):
            _x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(x)
            _x = nn.leaky_relu(_x)
            _x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(_x)
            _x = nn.leaky_relu(_x)
            results.append(
                nn.Dense(self.state_dim, kernel_init=init.kaiming_normal())(_x)
            )
        _x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(x)
        _x = nn.leaky_relu(_x)
        _x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(_x)
        _x = nn.leaky_relu(_x)
        reward = nn.Dense(1, kernel_init=init.kaiming_normal())(_x)
        return jnp.stack(results, axis=0), reward


def init_model(obs_space, action_space, hidden_size, lr, init_key):
    key, model_key = random.split(init_key)
    model = ModelNetwork(
        hidden_size=hidden_size,
        state_dim=obs_space.shape[0],
        action_dim=action_space.shape[0],
        ensemble_members=8,
    )
    key1, key2 = random.split(model_key)
    x = random.normal(
        key1, (obs_space.shape[0] + action_space.shape[0],)
    )  # Dummy input
    params = model.init(key2, x)["params"]  # Initialization call
    tx = optax.rmsprop(lr)
    optim_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return optim_state, model
