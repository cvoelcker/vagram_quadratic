import jax
from flax import linen as nn


class QNetwork(nn.Module):
    hidden_size: int
    state_dim: int
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

class PolicyNetwork(nn.Module):
    hidden_size: int
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.action_dim)(x)
        logvar = nn.Dense(self.action_dim)(x)
        return mu, logvar

