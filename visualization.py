import matplotlib.pyplot as plt
import jax
import jax.numpy as jumpy
import numpy as np


def plot_2d_loss(loss_function, target, x_bounds, y_bounds, granularity):
    x = jumpy.linspace(x_bounds[0], x_bounds[1], granularity)
    y = jumpy.linspace(y_bounds[0], y_bounds[1], granularity)
    X, Y = jumpy.meshgrid(x, y)
    inp = jumpy.stack([X, Y], axis=-1)
    loss = jax.vmap(loss_function, in_axes=(0, None))(inp, target)
    plt.contourf(X, Y, loss, cmap='RdBu')
    plt.colorbar()
    plt.show()

