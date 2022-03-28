import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state

import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from sac_agent import SACAgent

from sac_networks import QNetwork, PolicyNetwork
from utils import ReplayBuffer, Transition
from environment import make_pendulum, get_samples


def q_network_init(model, state_dim, action_dim, lr, key):
  key1, key2 = random.split(key)
  x = random.normal(key1, (state_dim + action_dim,)) # Dummy input
  params = model.init(key2, x)["params"] # Initialization call
  tx = optax.rmsprop(lr)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

def policy_network_init(model, state_dim, lr, key):
  key1, key2 = random.split(key)
  x = random.normal(key1, (state_dim,)) # Dummy input
  params = model.init(key2, x)["params"] # Initialization call
  tx = optax.rmsprop(lr)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

def build_networks(hidden_dim: int, state_dim: int, action_dim: int, lr: float, key):
    q_key_1, q_key_2, policy_key = random.split(key, num=3)
    q_network_1 = QNetwork(hidden_dim, state_dim, action_dim)
    q_1_train_state = q_network_init(q_network_1, state_dim, action_dim, lr, q_key_1)
    q_network_2 = QNetwork(hidden_dim, state_dim, action_dim)
    q_2_train_state = q_network_init(q_network_2, state_dim, action_dim, lr, q_key_2)
    policy_network = PolicyNetwork(hidden_dim, action_dim)
    policy_train_state = policy_network_init(policy_network, state_dim, lr, policy_key)
    return q_network_1, q_1_train_state, q_network_2, q_2_train_state, policy_network, policy_train_state

def train(env):
  # init PRNG
  key = random.PRNGKey(0)

  # hyperparameters
  hidden_dim = 64
  state_dim = 3
  action_dim = 1
  gamma = 0.9
  alpha = 0.7
  lr = 3e-4
  batch_size = 32
  n_samples = 1000
  total_env_steps = 0
  average_rewards = []
  eps = 1e-3

  replay_buffer = ReplayBuffer(2000)

  # init networks
  key, network_init_key = random.split(key)
  q_network_1, q_1_state, q_network_2, q_2_state, policy_network, policy_state = build_networks(hidden_dim, state_dim, action_dim, lr, network_init_key)
  agent = SACAgent(gamma, q_network_1, q_network_2, policy_network)

  def train_step(batch, action_dim, q_network_1, q_1_state, q_network_2, q_2_state, policy_network, policy_state):
    states = batch[0]
    rewards = batch[2]
    dones = batch[3]
    key = random.PRNGKey(1)
    key, a_sample_key = random.split(key)

    actions = agent.select_action(policy_state.params, a_sample_key, states, action_dim)

    targets = agent.compute_targets(q_1_state.params, q_2_state.params, rewards, dones, jnp.concatenate((states, actions), axis=-1), actions, alpha)
    
    @jax.jit
    def q_loss_function(p1, p2):
     return agent.compute_q_loss(p1, p2, jnp.concatenate((states, actions), axis=-1), targets)

    @jax.jit
    def policy_loss_function(p):
     return agent.compute_policy_loss(p, q_1_state.params, q_2_state.params, states, alpha, a_sample_key, action_dim)
    
    q_loss, q_grad = jax.jit(jax.value_and_grad(q_loss_function))(q_1_state.params, q_2_state.params)
    policy_loss, policy_grad = jax.jit(jax.value_and_grad(policy_loss_function))(policy_state.params)

    q_1_state = q_1_state.apply_gradients(grads=q_grad)
    q_2_state = q_2_state.apply_gradients(grads=q_grad)
    policy_state = policy_state.apply_gradients(grads=policy_grad)

    return q_1_state, q_2_state, policy_state, {"q": q_loss, "policy": policy_loss}

  # train loop

  def train_epoch(batch_size, rng, action_dim, total_env_steps, q_network_1, q_1_state, q_network_2, q_2_state, policy_network, policy_state, converged):
    """Train for a single epoch."""
    rng, z_rng = random.split(rng)
    steps_per_epoch = 1000

    observations, actions, rewards, dones, next_observations = get_samples(env, n=n_samples)

    for i in range(n_samples):
      obs = observations[i]
      action = actions[i]
      reward = rewards[i]
      done = dones[i]
      next_obs = next_observations[i]
      env_output = Transition(obs, action, reward, done, next_obs)
      replay_buffer.push(env_output)

    total_env_steps += n_samples

    for _ in tqdm(range(steps_per_epoch)):
      batch = replay_buffer.sample(batch_size)

      q_1_state, q_2_state, policy_state, metrics = train_step(batch, action_dim, q_network_1, q_1_state, q_network_2, q_2_state, policy_network, policy_state)

    # eval
    if total_env_steps % 10000 == 0:
      traj_num = 0
      eval_done = False
      total_rewards = []

      while traj_num < 20:
        eval_obs = env.reset()
        total_episode_rewards = []
        while not eval_done:
          key = random.PRNGKey(1)
          key, a_sample_eval_key = random.split(key)
          eval_action = agent.select_action(policy_state.params, a_sample_eval_key, eval_obs, action_dim)
          eval_obs, eval_reward, eval_done = env.step(eval_action)
          total_episode_rewards.append(eval_reward)
        traj_num += 1
        total_rewards.append(total_episode_rewards.mean())

      average_rewards.append(total_rewards.mean())

      if abs(average_rewards[-1] - average_rewards[-2]) < eps:
        converged = True

      plt.plot(average_rewards)
      plt.show()

    return average_rewards, converged

  all_average_rewards = []
  converged = False
  while not converged:
    key, epoch_key = random.split(key)
    average_rewards, converged = train_epoch(batch_size, epoch_key, action_dim, total_env_steps, q_network_1, q_1_state, q_network_2, q_2_state, policy_network, policy_state, converged)
    all_average_rewards.append(average_rewards)
  return all_average_rewards


if __name__ == "__main__":
  env = make_pendulum()
  all_average_rewards = train(env)