from functools import partial
from tqdm import tqdm

import jax
from jax import random
from jax import numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(4,))
def train_step(inp, rewards, next_states, model_state, loss_function, key):
    # loss = loss_function(model_state, states, actions, rewards, next_states, key)
    loss, grads = jax.value_and_grad(loss_function)(
        model_state.params, inp, rewards, next_states, key
    )
    model_state = model_state.apply_gradients(grads=grads)
    return loss, model_state


def train_model(model_state, replay_buffer, loss_function, batch_size, rng, patience=5):
    train, val = replay_buffer.train_validation_split()

    def train_epoch(model_state, rng):
        """Train for a single epoch."""
        train_ds_size = len(train.observations)
        steps_per_epoch = train_ds_size // batch_size

        rng, perm_key = random.split(rng)
        perms = random.permutation(perm_key, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        loss_values = []

        for perm in perms:
            # get batch from permutations
            states = train.observations[perm]
            actions = train.actions[perm]
            if len(actions.shape) == 1:
                actions = actions.reshape(-1, 1)
            rewards = train.rewards[perm].reshape(-1, 1)
            next_states = train.next_observations[perm]

            rng, step_key = random.split(rng)

            inp = jnp.concatenate((states, actions), axis=1)
            loss_value, model_state = train_step(
                inp,
                rewards,
                next_states,
                model_state,
                loss_function,
                step_key,
            )
            loss_values.append(loss_value)
        return model_state, jnp.array(np.array(loss_values))

    def validate(batch, model_state, loss_function, best_loss, validation_key):
        if len(batch.actions.shape) == 1:
            actions = batch.actions.reshape(-1, 1)
        else:
            actions = batch.actions
        inp = jnp.concatenate((batch.observations, actions), axis=1)
        loss = jax.jit(loss_function)(
            model_state.params,
            inp,
            batch.rewards.reshape(-1, 1),
            batch.next_observations,
            validation_key,
        )
        return loss, loss > best_loss

    converged = False
    all_losses = []
    not_improved = 0
    best_valid_loss = float("inf")
    while not converged:
        rng, epoch_key, validation_key = random.split(rng, 3)
        model_state, losses = train_epoch(model_state, epoch_key)
        all_losses.append(losses)
        val_loss, improved = validate(
            val, model_state, loss_function, best_valid_loss, validation_key
        )
        best_valid_loss = min(best_valid_loss, val_loss)
        if improved:
            not_improved += 1
        else:
            not_improved = 0
        if not_improved > patience:
            converged = True

    return model_state, jnp.concatenate(all_losses)
