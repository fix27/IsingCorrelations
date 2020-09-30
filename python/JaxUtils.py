import jax
from jax.experimental import stax


@jax.jit
def logcosh(x):
    x = x * jax.numpy.sign(x.real)
    return x + jax.numpy.log(1.0 + jax.numpy.exp(-2.0 * x)) - jax.numpy.log(2.0)


LogCoshLayer = stax.elementwise(logcosh)


def SumLayer():
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        return inputs.sum(axis=-1)

    return init_fun, apply_fun


SumLayer = SumLayer()


__all__ = ["LogCoshLayer", "SumLayer"]
