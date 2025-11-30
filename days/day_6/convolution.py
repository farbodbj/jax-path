from typing import Tuple
from functools import partial
from dataclasses import dataclass
import jax.numpy as jnp
import jax

class JaxModule:
    def fwd(self, X: jax.Array)->jax.Array:
        raise NotImplementedError(f"fwd function not implmented in {self.__qualname__}")

    def rev(self):
        return jax.grad(self.fwd)

class ConvLayer(JaxModule):
    def __init__(self, key: jax.Array, stride: int, filter_size: Tuple[int, int], padding: int):
        super().__init__()
        k1, k2 = jax.random.split(key)

        self.filter: jax.Array = jax.random.normal(k1, shape=(filter_size, filter_size))
        self.bias = jax.random.uniform(k2).item(0)

    @partial(jax.vmap, in_axes=(None, 0))
    def fwd(self, X: jax.Array)->jax.Array:
        filter_applied = jax.scipy.signal.convolve2d(X, self.filter, mode="valid")
        return jnp.ones_like(filter_applied) * self.bias + filter_applied

class LinearLayer(JaxModule):
    def __init__(self, key: jax.Array, inout: Tuple[int,int]):
        k1, k2 = jax.random.split(key)
        self.weights = jax.random.normal(k1, shape=inout)
        self.biases = jax.random.normal(k2, shape=(1, inout[1]))

    @partial(jax.vmap, in_axes=(None, 0))
    def fwd(self, X: jax.Array)->jax.Array:
        return X.T @ self.weights + self.biases

@dataclass
class FlattenLayer(JaxModule):
    @partial(jax.vmap, in_axes=(None, 0))
    def fwd(self, X: jax.Array):
        flattened = X.flatten()
        return flattened.reshape(flattened.shape[0], 1)



def sigmoid(X: jax.Array)->jax.Array:
    return 1 / (1 + jnp.exp(-X))
def relu(X: jax.Array)->jax.Array:
    return jnp.where(X > 0, X, 0)
def bce_loss(Y: jax.Array, Y_hat: jax.Array)->float:
    eps = 1e-7 # 1e-8 is too low
    Y_hat_cliped = jnp.clip(Y_hat, eps, 1-eps)
    return - (Y * jnp.log(Y_hat_cliped) + (1-Y) * jnp.log(1-Y_hat_cliped))


if __name__ == "__main__":
    master_key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(master_key, 4)
    
    N = 192
    X = jax.random.normal(key=k1, shape=(N, 8,8))
    Y = jnp.where(jax.random.uniform(key=k4, shape=(N)) > 0.5, 1.0, 0.0) 

    l1 = ConvLayer(key=k2, stride=1, filter_size=2, padding=1)
    Y_hat = l1.fwd(X)

    l2 = FlattenLayer()
    Y_hat = l2.fwd(Y_hat)

    l3 = LinearLayer(key=k3, inout=(Y_hat.shape[1], 1))
    Y_hat = sigmoid(l3.fwd(Y_hat)).reshape((N,))
    loss = bce_loss(Y = Y, Y_hat = Y_hat)

    print(f"loss: {loss}")