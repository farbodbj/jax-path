'''
Some activation functions are implemented using JAX and their performance is compared
'''

import jax
import jax.numpy as jnp
import time
from functools import partial
import numpy as np

class LRelu:
   @staticmethod
   @jax.jit
   def leaky_relu_jax(X, alpha):
      '''Branchless implementation'''
      return jnp.greater_equal(X, 0).astype(int) * jnp.maximum(X, jnp.zeros_like(X)) + jnp.less(X, 0).astype(int) * (alpha * X)

   @staticmethod
   @jax.jit
   def leaky_relu_jax_2(X, alpha):
      return jnp.where(X > 0, X, alpha * X)

   @staticmethod
   def leaky_relu_np(X, alpha):
      return np.greater_equal(X, 0).astype(int) * np.maximum(X, np.zeros_like(X)) + np.less(X, 0).astype(int) * (alpha * X)

class Sigmoid:
   @staticmethod
   @jax.jit
   def sigmoid_jax(X):
      return 1 / (1 + jnp.exp(-X))
   
   @staticmethod
   def sigmoid_np(X):
      return 1 / (1 + np.exp(-X))

class Selu:
   '''
      SELU(x)=scale*(max(0,x)+min(0,α*(exp(x)-1)))
      with α=1.6732632423543772848170429916717 and scale=1.0507009873554804934193349852946
   '''

   @staticmethod
   @partial(jax.jit, static_argnames=["scale", "alpha"])
   def selu_jax(X, scale, alpha):   
      return scale * (jnp.maximum(0, X) + jnp.minimum(0, alpha * (jnp.exp(X)-1)))
   
   @staticmethod
   def selu_np(X, scale, alpha):   
      return scale * (np.maximum(0, X) + np.minimum(0, alpha * (np.exp(X)-1)))

if __name__ == "__main__":
    k = jax.random.PRNGKey(128931823) # key smash :)
    arr = jax.random.uniform(k, shape=(8_000_000), minval=-1, maxval=1)

    print("Benchmarking LeakyRelu")

    s = time.perf_counter()
    LRelu.leaky_relu_jax(arr, -0.1)
    print(f"Jax V1, took: {(time.perf_counter() - s):.4f}", end="\n\n")


    s = time.perf_counter()
    LRelu.leaky_relu_jax_2(arr, -0.1)
    print(f"Jax v2, took: {(time.perf_counter() - s):.4f}", end="\n\n")


    s = time.perf_counter()
    LRelu.leaky_relu_np(arr, -0.1)
    print(f"LeakyRelu numpy, took: {(time.perf_counter() - s):.4f}", end="\n\n")

    print("===========================", end="\n\n")

    
    print("Benchmarking Sigmoid\n")

    s = time.perf_counter()
    Sigmoid.sigmoid_jax(arr)
    print(f"sigmoid jax, took: {(time.perf_counter() - s):.4f}", end="\n\n")

    s = time.perf_counter()
    Sigmoid.sigmoid_np(arr)
    print(f"sigmoid numpy, took: {(time.perf_counter() - s):.4f}", end="\n\n")


    print("===========================", end="\n\n")

    
    print("Benchmarking Selu\n")

    s = time.perf_counter()
    Selu.selu_jax(X=arr, scale=1.6732632, alpha=1.050700)
    print(f"selu jax, took: {(time.perf_counter() - s):.4f}", end="\n\n")

    s = time.perf_counter()
    Selu.selu_np(arr, scale=1.6732632, alpha=1.050700)
    print(f"selu numpy, took: {(time.perf_counter() - s):.4f}", end="\n\n")
    