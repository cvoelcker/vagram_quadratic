import jax
from flax import linen as nn


class QNetwork(nn.Module):
    hidden_size: int
    state_dim: int
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
                self.hidden_size,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)
        x = nn.relu(x)
        x = nn.Dense(
                self.hidden_size,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)
        x = nn.relu(x)
        return nn.Dense(
                1,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)

class PolicyNetwork(nn.Module):
    hidden_size: int
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
                self.hidden_size,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)
        x = nn.relu(x)
        x = nn.Dense(
                self.hidden_size,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)
        x = nn.relu(x)
        mu = nn.Dense(
                self.action_dim,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)
        logvar = nn.Dense(
                self.action_dim,
                kernel_init=jax.nn.initializers.he_normal()
                )(x)
        return mu, logvar

