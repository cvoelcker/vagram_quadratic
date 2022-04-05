import jax
import jax.numpy as jnp


def hessian(fun):
    """Compute the hessian for an input function"""
    return jax.jacfwd(jax.jacrev(fun))


def model_prediction(theta, model, inp):
    """Compute the model prediction for a given input."""
    return model.apply({"params": theta}, inp)


def mse_loss(model_prediction, environment_sample, unused, unused2, unused3):
    """Compute the MSE loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    return jnp.mean(jnp.square(err))


def vagram_loss(model_prediction, environment_sample, value_function, hess, key):
    """Compute the VAGRAM loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    _, grad = jax.value_and_grad(value_function)(environment_sample, key)
    return jnp.square(grad * err).sum()


def vagram_broken_loss(model_prediction, environment_sample, value_function, hess, key):
    """Compute the VAGRAM loss for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    _, grad = jax.value_and_grad(value_function)(environment_sample, key)
    return jnp.square(grad.dot(err)).sum()


def quadratic_vagram_broken_loss(
    model_prediction, environment_sample, value_function, hess, key
):
    """Compute the quadratic upper bounded 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hess(environment_sample, key)
    return jnp.square(err.T.dot(hes.dot(err)))


def quadratic_vagram_loss(
    model_prediction, environment_sample, value_function, hess, key
):
    """Compute the quadratic upper bounded 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hess(environment_sample, key)
    l, Q = jnp.linalg.eigh(hes)
    basis_trans = jnp.dot(jnp.transpose(Q), err)
    return jnp.square(l * basis_trans).sum()


def quadratic_vagram_loss_quartic(
    model_prediction, environment_sample, value_function, key
):
    """Compute the original quartic 2nd order VAGRAM loss using the Hessian for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    hes = hessian(value_function)(environment_sample, key)
    l, Q = jnp.linalg.eigh(hes)
    basis_trans = jnp.dot(jnp.transpose(Q), err)
    return jnp.square(l * jnp.square(basis_trans)).sum()


def eval_loss_vaml(model_prediction, environment_sample, value_function, key):
    """Compute the VAML loss for validation for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = value_function(environment_sample, key) - value_function(
        model_prediction, key
    )
    return jnp.square(err).sum()


def eval_loss_mse(model_prediction, environment_sample, value_function, unused):
    """Compute the MSE loss for validation for a given model prediction and environment sample. (VMAPed over batch for prediction and target)"""
    err = model_prediction - environment_sample
    return jnp.square(err).sum()


def make_loss(loss_fn, network, value_function):
    """
    Compile a loss function with a given network structure, loss, and value function
    """

    hess = jax.jit(hessian(value_function))

    def loss(params, states, actions, rewards, next_states, key):
        def v_loss_fn(pred, next_state, value_function, key):
            return jax.vmap(loss_fn, in_axes=[0, 0, None, None, None])(
                pred, next_state, value_function, hess, key
            )

        inp = jnp.concatenate([states, actions], axis=1)
        pred, pred_reward = network.apply({"params": params}, inp)
        losses = jax.vmap(v_loss_fn, in_axes=[0, None, None, None])(
            pred, next_states, value_function, key
        )
        reward_losses = mse_loss(pred_reward, rewards, value_function, None, None)
        return jnp.mean(losses) + reward_losses

    return jax.jit(loss)
