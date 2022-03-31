import gym

import jax.numpy as np
import jax


def make_pendulum():
    return gym.make("Pendulum-v1")


def get_samples(env, n=5000):
    obs = []
    actions = []
    rewards = []
    dones = []
    next_obs = []

    _obs = env.reset()
    obs.append([])
    actions.append([])
    rewards.append([])
    dones.append([])
    next_obs.append([])

    for i in range(n):
        obs[-1].append(_obs)
        action = env.action_space.sample()
        actions[-1].append(action)
        _obs, _reward, _done, _ = env.step(action)
        rewards[-1].append(_reward)
        dones[-1].append(_done)
        next_obs[-1].append(_obs)
        if _done:
            _obs = env.reset()
            obs.append([])
            actions.append([])
            rewards.append([])
            dones.append([])
            next_obs.append([])

    return np.array(obs[:-1]), np.array(actions[:-1]), np.array(rewards[:-1]), np.array(dones[:-1]), np.array(next_obs[:-1])
