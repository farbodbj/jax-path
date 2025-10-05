import jax
import jax.numpy as jnp
import time

@jax.vmap
@jax.jit
def eucleadian_dist(x, y):
    return jnp.sqrt(jnp.sum((x - y) ** 2))


if __name__ == "__main__":
    k = jax.random.PRNGKey(42)
    k1,k2,k3,k4 = jax.random.split(k, 4)

    arr_1 = jax.random.uniform(k1, shape=((1024, )), minval=-1, maxval=1).astype(float)
    arr_2 = jax.random.uniform(k2, shape=((1024, 8)), minval=-1, maxval=1).astype(float)


    s = time.perf_counter()
    eucleadian_dist(arr_1, arr_2)
    print(f"vmap, One pass took for: {time.perf_counter() - s:.4f}")


    arr_1 = jax.random.uniform(k3, shape=((1024, )), minval=-1, maxval=1).astype(float)
    arr_2 = jax.random.uniform(k4, shape=((1024, 8)), minval=-1, maxval=1).astype(float)

    s = time.perf_counter()
    eucleadian_dist(arr_1, arr_2)
    print(f"vmap + jit, One pass took for: {time.perf_counter() - s:.4f}")
