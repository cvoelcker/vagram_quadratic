import collections
import random
import numpy as np

Transition = collections.namedtuple("Transition", "obs reward done action next_obs")


# Adapted from: https://github.com/deepmind/rlax/blob/master/examples/simple_dqn.py
class ReplayBuffer(object):
  """A simple Python replay buffer."""

  def __init__(self, capacity):
    self.buffer = collections.deque(maxlen=capacity)

  def push(self, env_output):

    if env_output.action is not None:
      self.buffer.append(
          (env_output.obs, env_output.action, env_output.reward,
           env_output.next_obs, env_output.done))

  def sample(self, batch_size):
    obs_tm1, a_tm1, r_t, obs_t, done_t = zip(
        *random.sample(self.buffer, batch_size))
    return (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
           np.stack(obs_t), np.asarray(done_t))

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)