from copy import deepcopy
import gym
import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state

import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from sac_agent import (
    compute_policy_loss,
    compute_q_loss,
    compute_targets,
    select_action,
    select_mean_action,
)

from sac_networks import QNetwork, PolicyNetwork
from utils import ReplayBuffer, Transition


@jax.jit
def update(params1, params2, rho):
    new_params = jax.tree_multimap(
        lambda param1, param2: param1 * rho + (1 - rho) * param2, params1, params2
    )
    return new_params


def q_network_init(env, model, lr, key):
    key1, key2 = random.split(key)
    x = jnp.concatenate((env.observation_space.sample(), env.action_space.sample()), axis=-1)
    params = model.init(key2, x)["params"]  # Initialization call
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def policy_network_init(env, model, lr, key):
    key1, key2 = random.split(key)
    x = env.observation_space.sample()
    params = model.init(key2, x)["params"]  # Initialization call
    tx = optax.adam(lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def build_networks(env, hidden_dim: int, state_dim: int, action_dim: int, lr: float, key):
    q_key_1, q_key_2, policy_key = random.split(key, num=3)
    q_network = QNetwork(hidden_dim, state_dim, action_dim)
    q_1_state = q_network_init(env, q_network, lr, q_key_1)
    q_2_state = q_network_init(env, q_network, lr, q_key_2)
    q_t_1_state = deepcopy(q_1_state)
    q_t_2_state = deepcopy(q_2_state)
    policy_network = PolicyNetwork(hidden_dim, action_dim)
    policy_train_state = policy_network_init(env, policy_network, lr, policy_key)
    return (
        q_network,
        q_1_state,
        q_2_state,
        q_t_1_state,
        q_t_2_state,
        policy_network,
        policy_train_state,
    )


def train(env):
    # init PRNG
    key = random.PRNGKey(0)

    # hyperparameters
    hidden_dim = 128
    state_dim = 3
    action_dim = 1
    gamma = 0.99
    alpha = 0.7
    lr = 3e-4
    batch_size = 128
    n_samples = 1000
    total_env_steps = 0
    average_rewards = []
    rho = 0.995
    act_limit = env.action_space.high[0]
    observation = env.reset()

    replay_buffer = ReplayBuffer(2000)

    # init networks
    key, network_init_key = random.split(key)
    (
        q_network,
        q_1_state,
        q_2_state,
        q_t_1_state,
        q_t_2_state,
        policy_network,
        policy_state,
    ) = build_networks(env, hidden_dim, state_dim, action_dim, lr, network_init_key)

    @jax.jit
    def train_step(
        batch,
        key1,
        key2,
        q_1_state,
        q_2_state,
        q_t_1_state,
        q_t_2_state,
        policy_state,
    ):
        states = batch[0]
        actions = batch[1]
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        rewards = batch[2].reshape(-1, 1)
        next_states = batch[3]
        dones = batch[4].reshape(-1, 1)

        targets = compute_targets(
            q_network,
            policy_network,
            q_t_1_state.params,
            q_t_2_state.params,
            policy_state.params,
            rewards,
            dones,
            next_states,
            key1,
            alpha,
            gamma,
            act_limit
        )

        def q_loss_function(p):
            return compute_q_loss(
                q_network,
                p,
                jnp.concatenate((states, actions), axis=-1),
                targets,
            )

        def policy_loss_function(p):
            return compute_policy_loss(
                policy_network,
                q_network,
                p,
                q_1_state.params,
                q_2_state.params,
                states,
                alpha,
                key2,
                act_limit
            )

        q1_loss, q1_grad = jax.value_and_grad(q_loss_function)(
            q_1_state.params
        )
        q2_loss, q2_grad = jax.value_and_grad(q_loss_function)(
            q_2_state.params
        )
        policy_loss, policy_grad = jax.value_and_grad(policy_loss_function)(
            policy_state.params
        )

        q_1_state = q_1_state.apply_gradients(grads=q1_grad)
        q_2_state = q_2_state.apply_gradients(grads=q2_grad)
        policy_state = policy_state.apply_gradients(grads=policy_grad)

        q_t_1_state = q_t_1_state.replace(params=update(q_t_1_state.params, q_1_state.params, rho))
        q_t_2_state = q_t_2_state.replace(params=update(q_t_2_state.params, q_2_state.params, rho))

        return (
            q_1_state,
            q_2_state,
            q_t_1_state,
            q_t_2_state,
            policy_state,
            {"q": (q1_loss + q2_loss).mean(), "policy": policy_loss},
        )

    # train loop

    def train_epoch(
        batch_size,
        rng,
        total_env_steps,
        q_network,
        q_1_state,
        q_2_state,
        q_t_1_state,
        q_t_2_state,
        policy_network,
        policy_state,
        converged,
        observation
    ):
        """Train for a single epoch."""
        steps_per_epoch = 1000
        for _ in range(n_samples):
            if total_env_steps < n_samples:
              action = env.action_space.sample()
            else: 
              rng, a_rng = random.split(rng)
              action, _ = select_action(policy_network, policy_state.params, a_rng, observation, act_limit)

            next_obs, reward, done, _ = env.step(action)
            env_output = Transition(jnp.array([observation]), jnp.array([action]), jnp.array([reward]), jnp.array([done]), jnp.array([next_obs]))
            replay_buffer.push(env_output)
            if not done:
              observation = next_obs
            else:
              observation = env.reset()

        total_env_steps += n_samples

        batch_sampler = replay_buffer.sample(batch_size)
        for _ in tqdm(range(steps_per_epoch)):
            rng, key1, key2, batch_rng = random.split(rng, num=4)
            batch = batch_sampler(batch_rng)
            (
                q_1_state,
                q_2_state,
                q_t_1_state,
                q_t_2_state,
                policy_state,
                metrics,
            ) = train_step(
                batch,
                key1,
                key2,
                q_1_state,
                q_2_state,
                q_t_1_state,
                q_t_2_state,
                policy_state,
            )
        print(metrics)

        # eval
        def gather_eval_trajectory():
            @jax.jit
            def action(obs):
                return select_mean_action(policy_network, policy_state.params, obs)

            eval_done = False
            eval_obs = env.reset()
            total_episode_rewards = 0
            while not eval_done:
                eval_obs = jax.device_put(eval_obs)
                eval_action = action(eval_obs)
                eval_obs, eval_reward, eval_done, _ = env.step(eval_action)
                total_episode_rewards += eval_reward
            return total_episode_rewards

        average_rewards = 0
        for i in tqdm(range(20)):
            average_rewards += gather_eval_trajectory()
        average_rewards /= 20

        print(average_rewards)

        return average_rewards, total_env_steps, converged

    all_average_rewards = []
    converged = False
    while not converged:
        key, epoch_key = random.split(key)
        average_rewards, total_env_steps, converged = train_epoch(
            batch_size,
            epoch_key,
            total_env_steps,
            q_network,
            q_1_state,
            q_2_state,
            q_t_1_state,
            q_t_2_state,
            policy_network,
            policy_state,
            converged,
            observation,
        )
        all_average_rewards.append(average_rewards)
    return all_average_rewards


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    all_average_rewards = train(env)
