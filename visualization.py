import jax
import jax.numpy as jumpy
from main import (
    eval_loss_vaml,
    mse_loss,
    quadratic_vagram_broken_loss,
    quadratic_vagram_loss,
    vagram_broken_loss,
    vagram_loss,
    vagram_no_bounds_loss,
)
import numpy as np

import seaborn as sns
import matplotlib as mpl

sns.set()
mpl.rcParams["xtick.labelsize"] = 30
mpl.rcParams["ytick.labelsize"] = 30
mpl.rcParams["axes.titlesize"] = 40
mpl.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt


def plot_2d_loss(
    ax, loss_function, target, x_bounds, y_bounds, granularity, cmap="viridis"
):
    x = jumpy.linspace(x_bounds[0], x_bounds[1], granularity)
    y = jumpy.linspace(y_bounds[0], y_bounds[1], granularity)
    X, Y = jumpy.meshgrid(x, y)
    inp = jumpy.stack([X, Y], axis=-1)
    inp = inp.reshape(-1, 2)

    loss = jax.vmap(loss_function, in_axes=(0, None))(inp, target)

    loss = loss.reshape(granularity, granularity)
    cont = ax.contourf(X, Y, loss, levels=30, cmap=cmap)
    ax.scatter(target[0], target[1])
    ax.set_aspect("equal", "box")


if __name__ == "__main__":

    def value_function(x):
        return jumpy.sin(x[0]) * jumpy.sin(x[1])

    target = jumpy.array([1.0, 0.5])

    fig, axs = plt.subplots(2, 4, figsize=(60, 30))

    plot_2d_loss(
        axs[0, 0],
        lambda x, y: value_function(x),
        target,
        [-2, 2],
        [-2, 2],
        100,
        cmap="inferno",
    )
    axs[0, 0].set_title("Value function")
    plot_2d_loss(
        axs[0, 1],
        lambda x, y: mse_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[0, 1].set_title("MSE")
    plot_2d_loss(
        axs[0, 2],
        lambda x, y: vagram_broken_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[0, 2].set_title("VAGRAM unbounded")
    plot_2d_loss(
        axs[0, 3],
        lambda x, y: vagram_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[0, 3].set_title("VAGRAM")
    plot_2d_loss(
        axs[1, 0],
        lambda x, y: quadratic_vagram_broken_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[1, 0].set_title("Quadratic VAGRAM unbounded")
    plot_2d_loss(
        axs[1, 1],
        lambda x, y: quadratic_vagram_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[1, 1].set_title("Quadratic VAGRAM")
    plot_2d_loss(
        axs[1, 2],
        lambda x, y: vagram_no_bounds_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[1, 2].set_title("VAGRAM + Quadratic VAGRAM unbounded")
    plot_2d_loss(
        axs[1, 3],
        lambda x, y: 2 * vagram_loss(x, y, value_function)
        + 0.5 * quadratic_vagram_loss(x, y, value_function),
        target,
        [-2, 2],
        [-2, 2],
        100,
    )
    axs[1, 3].set_title("VAGRAM + Quadratic VAGRAM")
    plt.savefig(f"plt/loss_functions_{target}.png", bbox_inches="tight")

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
