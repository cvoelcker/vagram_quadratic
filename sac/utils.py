import collections
import jax
import jax.numpy as jnp
from jax import random

Transition = collections.namedtuple("Transition", "obs action reward done next_obs")


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
        self._obs = jnp.concatenate(self.observations)
        self._next_obs = jnp.concatenate(self.next_observations)
        self._rewards = jnp.concatenate(self.rewards)
        self._actions = jnp.concatenate(self.actions)
        self._dones = jnp.concatenate(self.dones)
        self._compiled = True
        self.counter = len(self._obs)
        print(self.counter)

    def sample(self, batch_size):
        if not self._compiled:
            self._compile()

        @jax.jit
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

        return sampler
