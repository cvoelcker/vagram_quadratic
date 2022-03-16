import gym

import jax.numpy as np
import jax


def make_pendulum():
    return gym.make("Pendulum-v1")


def get_samples(env, n=5000):
    obs = []
    actions = []

    _obs = env.reset()
    obs.append([])
    actions.append([])

    for i in range(n):
        obs[-1].append(_obs)
        action = env.action_space.sample()
        actions[-1].append(action)
        _obs, _, done, _ = env.step(action)
        if done:
            _obs = env.reset()
            obs.append([])
            actions.append([])

    return np.array(obs[:-1]), np.array(actions[:-1])
