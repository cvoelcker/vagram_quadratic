import jax
from jax import random
from jax import numpy as jnp


def train_step(states, actions, rewards, next_states, model_state, loss_function):
    loss, grads = jax.value_and_grad(loss_function)(
        model_state,
        states,
        actions,
        rewards,
        next_states,
    )
    model_state = model_state.apply_gradient(grads)
    return loss, model_state


def train_model(model_state, replay_buffer, loss_function, batch_size, rng, patience=5):
    train, val = replay_buffer.train_validation_split()

    def train_epoch(model_state, rng):
        """Train for a single epoch."""
        train_ds_size = len(train.states)
        steps_per_epoch = train_ds_size // batch_size

        perms = random.permutation(rng, train_ds_size)
        perms = perms[: steps_per_epoch * batch_size]
        perms = perms.reshape((steps_per_epoch, batch_size))
        loss_values = []

        for perm in perms:
            # get batch from permutations
            states = train.states[perm]
            actions = train.actions[perm]
            rewards = train.rewards[perm]
            next_states = train.next_states[perm]

            @jax.jit
            def f(states, actions, rewards, next_states, model_state):
                return train_step(
                    states, actions, rewards, next_states, model_state, loss_function
                )

            loss_value, model_state = f(
                states, actions, rewards, next_states, model_state
            )
            loss_values.append(loss_value)
        return model_state, loss_values

    def validate(batch, model_state, loss_function, best_loss):
        loss = loss_function(
            batch.states, batch.actions, batch.next_states, model_state
        )
        return loss > best_loss

    converged = False
    all_losses = []
    not_improved = 0
    best_valid_loss = float("inf")
    while not converged:
        rng, epoch_key = random.split(rng)
        model_state, losses = train_epoch(model_state, epoch_key)
        if validate(val, model_state, loss_function, best_valid_loss):
            not_improved += 1
        if not_improved > patience:
            converged = True

    return model_state, jnp.concatenate(all_losses)
