import matplotlib.pyplot as plt
import jax
import jax.numpy as jumpy
from main import (
    mse_loss,
    quadratic_vagram_broken_loss,
    quadratic_vagram_loss,
    vagram_broken_loss,
    vagram_loss,
)
import numpy as np


def plot_2d_loss(loss_function, target, x_bounds, y_bounds, granularity):
    x = jumpy.linspace(x_bounds[0], x_bounds[1], granularity)
    y = jumpy.linspace(y_bounds[0], y_bounds[1], granularity)
    X, Y = jumpy.meshgrid(x, y)
    inp = jumpy.stack([X, Y], axis=-1)
    inp = inp.reshape(-1, 2)

    loss = jax.vmap(loss_function, in_axes=(0, None))(inp, target)

    loss = jumpy.log(loss.reshape(granularity, granularity))
    plt.contourf(X, Y, loss, cmap="RdBu")
    plt.scatter(target[0], target[1])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":

    def value_function(x):
        return jumpy.sin(x[0]) * jumpy.cos(x[1])

    def l(x, y):
        return quadratic_vagram_broken_loss(x, y, value_function) + vagram_broken_loss(
            x, y, value_function
        )

    plot_2d_loss(l, jumpy.array([1.0, 2.0]), [-10, 10], [-10, 10], 100)

    # def value_function(x):
    #     return jumpy.prod(jumpy.sin(x))

    # def l(x, y):
    #     return vagram_loss(x, y, value_function)

    # plot_2d_loss(l, jumpy.array([1.0, 2.0]), [-4, 4], [-4, 4], 100)

    # def value_function(x):
    #     return jumpy.prod(jumpy.sin(x))

    # def l(x, y):
    #     return mse_loss(x, y, value_function)

    # plot_2d_loss(l, jumpy.array([1.0, 2.0]), [-4, 4], [-4, 4], 100)
