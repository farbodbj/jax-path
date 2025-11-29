
import jax.numpy as jnp
from jax import jacfwd,jacrev,jacobian,hessian
from time import perf_counter_ns
import jax



def single_input_jacobian():
    def f(x):
        # toy function with one input
        return (x / 2) + jnp.cos(x * 2)

    x = 0.03
    f_jacobian_fwd = jacfwd(f)
    f_jacobian_rev = jacrev(f)

    then = perf_counter_ns()
    print(f_jacobian_fwd(x))
    print(f"f_jacobian_fwd took: {(perf_counter_ns() - then) / 1000000}")
    then = perf_counter_ns()
    print(f_jacobian_rev(x))
    print(f"f_jacobian_rev took: {(perf_counter_ns() - then) / 1000000}\n\n")


def multi_input_jacobian():
    def f_multi_inupt(x1, x2, x3, x4, x5, x6):
        return jnp.cos(x1 * x2) * jnp.sin(x3 * x4) * jnp.atan(x5 * x6)

    k = jax.random.PRNGKey(42)    
    k1,k2,k3,k4,k5,k6 = jax.random.split(k, 6)
    keys = [k1,k2,k3,k4,k5,k6]
    x1,x2,x3,x4,x5,x6 = [jax.random.normal(k, shape=(1,)) for k in keys]

    f_multi_input_jacobian_fwd = jacfwd(f_multi_inupt)
    f_multi_input_jacobian_rev = jacrev(f_multi_inupt)

    then = perf_counter_ns()
    print(f_multi_input_jacobian_fwd(x1,x2,x3,x4,x5,x6))
    # as stated in the documents (https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)  the jacfwd has an edge over the jacrev for 
    # tall jacobian matrices, that is when the function has more inputs than outputs
    print(f"f_multi_input_jacobian_fwd took: {(perf_counter_ns() - then) / 1000000}") 
    then = perf_counter_ns()
    print(f_multi_input_jacobian_rev(x1,x2,x3,x4,x5,x6))
    print(f"f_multi_input_jacobian_rev took: {(perf_counter_ns() - then) / 1000000}")


if __name__ == "__main__":
    # Read this great article on jacobian and hessian matrix, just a reminder of Math II.
    single_input_jacobian()
    multi_input_jacobian()
