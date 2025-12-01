from typing import Tuple, List, Dict
from dataclasses import dataclass
from functools import partial
import jax.numpy as jnp
import jax

class JaxModule:
    params: jax.Array
    
    def __call__(self, X: jax.Array):
        return self.fwd(self.params, X)
        
    def fwd(self, params: jax.Array, X: jax.Array)->jax.Array:
        raise NotImplementedError(f"fwd function not implmented in {self.__qualname__}")

    def rev(self):
        return jax.grad(self.fwd)

class ConvLayer(JaxModule):
    def __init__(self, key: jax.Array, stride: int, filter_size: Tuple[int, int], padding: int):
        super().__init__()
        k1, k2 = jax.random.split(key)
        self.params = {
            "W": jax.random.normal(k1, shape=(filter_size, filter_size)),
            "b": jax.random.uniform(k2).item(0)
        }

    @partial(jax.vmap, in_axes=(None, None, 0))
    def fwd(self, params: jax.Array, X: jax.Array)->jax.Array:
        filter_applied = jax.scipy.signal.convolve2d(X, params["W"], mode="valid")
        return jnp.ones_like(filter_applied) * params["b"] + filter_applied


class LinearLayer(JaxModule):
    def __init__(self, key: jax.Array, inout: Tuple[int,int]):
        k1, k2 = jax.random.split(key)
        self.params = {
            "W": jax.random.normal(k1, shape=inout),
            "b": jax.random.normal(k2, shape=(1, inout[1]))
        }

    @partial(jax.vmap, in_axes=(None, None, 0))
    def fwd(self, params: jax.Array, X: jax.Array)->jax.Array:
        return X.T @ params["W"] + params["b"]

class FlattenLayer(JaxModule):
    def __init__(self):
        self.params = {}

    @partial(jax.vmap, in_axes=(None, None, 0))
    def fwd(self, params: jax.Array, X: jax.Array)->jax.Array:
        flattened = X.flatten()
        return flattened.reshape(flattened.shape[0], 1)

class ReLU(JaxModule):
    def __init__(self):
        self.params = {}

    def fwd(self, params: jax.Array, X: jax.Array):
        return jnp.where(X > 0, X, 0)

class Sigmoid(JaxModule):
    def __init__(self):
        self.params = {}

    def fwd(self, params: jax.Array, X: jax.Array):
        return 1 / (1 + jnp.exp(-X))

@dataclass
class SGDOptimzer:
    lr: float

    def sgd_fn(self, grad, x_old):
        return x_old - self.lr * grad
    
    def update_state(self, grads: Dict, old_state: Dict):
        return jax.tree.map(self.sgd_fn, grads, old_state)

class MyModel(JaxModule):
    def __init__(self, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.layers = [
            ConvLayer(key=k1, stride=1, filter_size=2, padding=1),
            FlattenLayer(),
            LinearLayer(key=k2, inout=(49, 1)), # hardcoded till i solve the autodiff problem
            Sigmoid()
        ]
        
        self.params = {}
        for idx, layer in enumerate(self.layers):
            self.params[self._get_layer_name(idx)] = layer.params

    def _get_layer_name(self, idx: int):
        return f"layer_{idx}"

    def fwd(self, params: Dict, X: jax.Array)->jax.Array:
        Y = None
        for idx, l in enumerate(self.layers): # the loop can be simpler but I chose this more readable style 
            if idx == 0:
                Y = l.fwd(params[self._get_layer_name(idx)], X)
            else:
                Y = l.fwd(params[self._get_layer_name(idx)], Y)
        return Y
        

def bce_loss(Y: jax.Array, Y_hat: jax.Array)->float:
    eps = 1e-7 # 1e-8 is too low
    
    Y_hat_cliped = jnp.clip(Y_hat, eps, 1-eps)
    Y = Y.reshape(Y_hat_cliped.shape)

    return - (Y * jnp.log(Y_hat_cliped) + (1 - Y) * jnp.log(1 - Y_hat_cliped))


if __name__ == "__main__":
    master_key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(master_key, 3)
    
    N = 192
    X = jax.random.normal(key=k1, shape=(N, 8,8))
    Y = jnp.where(jax.random.uniform(key=k2, shape=(N)) > 0.5, 1.0, 0.0)

    model = MyModel(key=k3)

    def loss_fn(params, X: jax.Array, Y: jax.Array):
        Y_hat = model.fwd(params, X)
        sample_loss = bce_loss(Y = Y, Y_hat = Y_hat)
        return jnp.mean(sample_loss)
    
    optim = SGDOptimzer(1e-2)
    for i in range(24):
        loss, grads = jax.value_and_grad(loss_fn, argnums=(0))(model.params, X, Y)
        model.params = optim.update_state(grads, model.params)
        print(f"loss: {loss}")