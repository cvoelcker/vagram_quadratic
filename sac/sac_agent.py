import jax
from jax import random
import jax.numpy as jnp


def pdf_normal(x, mu, sigma):
  return (1 / sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((x - mu)/sigma) ** 2)


def select_action(policy_network, theta, key, state):

    eps = 1e-6
    mu, sigma = policy_network.apply({"params": theta}, state)
    a_tilde = mu + sigma * random.normal(key, shape=mu.shape)
    gauss_log_prob = pdf_normal(a_tilde, mu, sigma)
    gauss_log_prob -= jnp.sum(jnp.log(jax.nn.relu(1 - jnp.tanh(a_tilde) ** 2) + eps), axis=1)
    return a_tilde, gauss_log_prob


@jax.jit
def select_mean_action(policy_network, theta, state):
    mu, _ = policy_network.apply({"params": theta}, state)
    return mu


def min_q_network(q_network, phi1, phi2, inp):
    return jax.lax.min(
        q_network.apply({"params": phi1}, inp),
        q_network.apply({"params": phi2}, inp),
    )


def compute_targets(q_network, policy_network, phi1, phi2, theta, reward, done, state, key, alpha, gamma):

    a_tilde, gauss_log_prob = select_action(policy_network, theta, key, state)
    return (
        reward
        + gamma * (1 - done) * min_q_network(q_network, phi1, phi2, jnp.concatenate((state, a_tilde), axis=-1))
        - alpha * gauss_log_prob
    )


def compute_q_loss(q_network, phi1, phi2, inp, target):
    return jnp.mean(
        (q_network.apply({"params": phi1}, inp) - target) ** 2
        + (q_network.apply({"params": phi2}, inp) - target) ** 2
    )


def compute_policy_loss(
    policy_network, q_network, theta, phi1, phi2, state, alpha, key
):
    
    a_tilde, gauss_log_prob = select_action(policy_network, theta, key, state)
    return jnp.mean(
        min_q_network(q_network, phi1, phi2, jnp.concatenate((state, a_tilde), axis=-1))
        - alpha * gauss_log_prob
    )
