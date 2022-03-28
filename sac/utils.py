import collections
import jax
import jax.numpy as jnp
from jax import random

Transition = collections.namedtuple("Transition", "obs reward done action next_obs")


# Adapted from: https://github.com/deepmind/rlax/blob/master/examples/simple_dqn.py
class ReplayBuffer(object):
    """A simple Python replay buffer."""

    def __init__(self, capacity):
        self.counter = 0
        self.capacity = capacity

        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

        self._compiled = False

    def push(self, env_output):
        self.observations.append(env_output.obs)
        self.actions.append(env_output.action)
        self.rewards.append(env_output.reward)
        self.next_observations.append(env_output.next_obs)
        self.dones.append(env_output.done)
        self._compiled = False

    def _compile(self):
        self._obs = jnp.stack(self.observations)
        self._next_obs = jnp.stack(self.next_observations)
        self._rewards = jnp.stack(self.rewards)
        self._actions = jnp.stack(self.actions)
        self._dones = jnp.stack(self.dones)
        self._compiled = True

    def sample(self, batch_size):
        if not self._compiled:
            self._compile()

        def sampler(key):
            perms = random.randint(
                key, shape=(batch_size,), minval=0, maxval=self.counter
            )
            return (
                self._obs[perms],
                self._actions[perms],
                self._rewards[perms],
                self._next_obs[perms],
                self._dones[perms],
            )

        return jax.jit(sampler)
