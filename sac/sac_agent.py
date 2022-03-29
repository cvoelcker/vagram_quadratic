import jax
from jax import random
import jax.numpy as jnp


class SACAgent:
    def __init__(self, gamma, q_network, policy_network):
        self.gamma = gamma
        self.q_network_1 = q_network
        self.q_network_2 = q_network
        self.q_target_1 = q_network
        self.q_target_2 = q_network
        self.policy_network = policy_network

    def select_action(self, theta, key, state, action_dim):
        mu, sigma = self.policy_network.apply({"params": theta}, state)
        action = mu + sigma * random.normal(key, shape=mu.shape)
        return action

    def min_q_network(self, phi1, phi2, inp):
        return jax.lax.min(
            self.q_network_1.apply({"params": phi1}, inp),
            self.q_network_2.apply({"params": phi2}, inp),
        )

    def min_q_target(self, phi1, phi2, inp):
        return jax.lax.min(
            self.q_target_1.apply({"params": phi1}, inp),
            self.q_target_2.apply({"params": phi2}, inp),
        )

    def compute_targets(self, phi1, phi2, reward, done, inp, action, alpha):
        return (
            reward
            + self.gamma * (1 - done) * self.min_q_network(phi1, phi2, inp)
            - alpha * jnp.log(action)
        )

    def compute_q_loss(self, phi1, phi2, inp, target):
        return jnp.mean(
            (self.q_network_1.apply({"params": phi1}, inp) - target) ** 2
            + (self.q_network_2.apply({"params": phi2}, inp) - target) ** 2
        )

    def compute_policy_loss(self, theta, phi1, phi2, state, alpha, key, action_dim):
        a_tilde = self.select_action(theta, key, state, action_dim)
        return jnp.mean(
            self.min_q_network(phi1, phi2, jnp.concatenate((state, a_tilde), axis=-1))
            - alpha * jnp.log(a_tilde)
        )


def select_action(policy_network, theta, key, state):
    mu, sigma = policy_network.apply({"params": theta}, state)
    action = mu + sigma * random.normal(key, shape=mu.shape)
    return action


def select_mean_action(policy_network, theta, state):
    mu, _ = policy_network.apply({"params": theta}, state)
    return mu


def min_q_network(q_network, phi1, phi2, inp):
    return jax.lax.min(
        q_network.apply({"params": phi1}, inp),
        q_network.apply({"params": phi2}, inp),
    )


def compute_targets(q_network, phi1, phi2, reward, done, inp, action, alpha, gamma):
    return (
        reward
        + gamma * (1 - done) * min_q_network(q_network, phi1, phi2, inp)
        - alpha * jnp.log(action)
    )


def compute_q_loss(q_network, phi1, phi2, inp, target):
    return jnp.mean(
        (q_network.apply({"params": phi1}, inp) - target) ** 2
        + (q_network.apply({"params": phi2}, inp) - target) ** 2
    )


def compute_policy_loss(
    policy_network, q_network, theta, phi1, phi2, state, alpha, key
):
    a_tilde = select_action(policy_network, theta, key, state)
    return jnp.mean(
        min_q_network(q_network, phi1, phi2, jnp.concatenate((state, a_tilde), axis=-1))
        - alpha * jnp.log(a_tilde)
    )
