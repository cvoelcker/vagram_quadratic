from flax import linen as nn
from jax.nn import initializers as init


class QNetwork(nn.Module):
    hidden_size: int
    state_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(x)
        x = nn.leaky_relu(x)
        return nn.Dense(1, kernel_init=init.kaiming_normal())(x)


class PolicyNetwork(nn.Module):
    hidden_size: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=init.kaiming_normal())(x)
        x = nn.leaky_relu(x)
        mu = nn.Dense(self.action_dim, kernel_init=init.kaiming_normal())(x)
        logvar = nn.Dense(self.action_dim, kernel_init=init.kaiming_normal())(x)
        return mu, logvar
