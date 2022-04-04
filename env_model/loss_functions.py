import jax
import jax.numpy as jnp


def hessian(fun):
    """Compute the hessian for an input function"""
    return jax.jit(jax.jacfwd(jax.jacrev(fun)))


def model_prediction(theta, model, inp):
    """Compute the model prediction for a given input."""
    return model.apply({"params": theta}, inp)


def mse_loss(model_prediction, environment_sample, unused):
    """Compute the MSE loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    return jnp.mean(jnp.square(err))


def vagram_loss(model_prediction, environment_sample, value_function):
    """Compute the VAGRAM loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    _, grad = jax.value_and_grad(value_function)(environment_sample)
    return jnp.square(grad * err).sum()


def vagram_broken_loss(model_prediction, environment_sample, value_function):
    """Compute the VAGRAM loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    _, grad = jax.value_and_grad(value_function)(environment_sample)
    return jnp.square(grad.dot(err)).sum()


def quadratic_vagram_broken_loss(model_prediction, environment_sample, value_function):
    """Compute the quadratic upper bounded 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample)
    print(hes)
    return jnp.square(err.T.dot(hes.dot(err)))


def quadratic_vagram_loss(model_prediction, environment_sample, value_function):
    """Compute the quadratic upper bounded 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample)
    l, Q = jnp.linalg.eigh(hes)
    basis_trans = jnp.dot(jnp.transpose(Q), err)
    print(l)
    return jnp.square(l * basis_trans).sum()


def quadratic_vagram_loss_quartic(model_prediction, environment_sample, value_function):
    """Compute the original quartic 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample)
    l, Q = jnp.linalg.eigh(hes)
    basis_trans = jnp.dot(jnp.transpose(Q), err)
    return jnp.square(l * jnp.square(basis_trans)).sum()


def eval_loss_vaml(model_prediction, environment_sample, value_function):
    """Compute the VAML loss for validation for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = value_function(environment_sample) - value_function(model_prediction)
    return jnp.square(err).sum()


def eval_loss_mse(model_prediction, environment_sample, value_function):
    """Compute the MSE loss for validation for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    return jnp.square(err).sum()


def make_loss(loss_fn, value_function):
    """
    Compile a loss function with a given network structure, loss, and value function
    """

    def loss(model_state, states, actions, rewards, next_states):
        inp = jnp.concatenate([states, actions], axis=1)
        pred, pred_reward = model_state.apply_fn({"params": model_state.params}, inp)
        losses = jax.vmap(loss_fn, in_axes=[0, None, None])(
            pred, next_states, value_function
        )
        reward_losses = mse_loss(pred_reward, rewards, value_function)
        return jnp.mean(losses) + reward_losses

    return jax.jit(loss)
