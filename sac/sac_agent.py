import jax
from jax import random
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def pdf_normal(x, mu, sigma):
  return (1 / sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * ((x - mu)/sigma) ** 2)


def select_action(policy_network, theta, key, state, act_limit):
    mu, log_std = policy_network.apply({"params": theta}, state)
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    dist = tfd.Normal(loc=mu, scale=jnp.exp(log_std))
    action = dist.sample(seed=key)
    log_prob_a = dist.log_prob(action)
    log_prob_a -= (2*(jnp.log(2) - action - jax.nn.softplus(-2*action)))
    action = jnp.tanh(action)
    action = act_limit * action
    return action, log_prob_a


def select_mean_action(policy_network, theta, state):
    mu, _ = policy_network.apply({"params": theta}, state)
    return jnp.tanh(mu)


def min_q_network(q_network, phi1, phi2, inp):
    return jax.lax.min(
        q_network.apply({"params": phi1}, inp),
        q_network.apply({"params": phi2}, inp),
    )


def compute_targets(q_network, policy_network, phi1, phi2, theta, reward, done, state, key, alpha, gamma, act_limit):
    a_tilde, gauss_log_prob = select_action(policy_network, theta, key, state, act_limit)
    target = (
        reward
        + gamma * (1 - done) * min_q_network(q_network, phi1, phi2, jnp.concatenate((state, a_tilde), axis=-1))
        - alpha * gauss_log_prob
    )
    return target


def compute_q_loss(q_network, phi, inp, target):
    losses = (
        (q_network.apply({"params": phi}, inp) - target) ** 2
    )
    # print()
    # print()
    # print()
    # print()
    # print(losses[:5])
    # print(q_network.apply({"params": phi1}, inp)[:5])
    # print(target[:5])
    return jnp.mean(losses)


def compute_policy_loss(
    policy_network, q_network, theta, phi1, phi2, state, alpha, key, act_limit
):
    
    a_tilde, gauss_log_prob = select_action(policy_network, theta, key, state, act_limit)
    losses = min_q_network(q_network, phi1, phi2, jnp.concatenate((state, a_tilde), axis=-1)) - alpha * gauss_log_prob

    # Change of sign because it'll be used for gradient ascent.
    return -jnp.mean(losses)
