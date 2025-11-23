from time import perf_counter_ns
import jax.lax as lax
import jax.numpy as jnp
import jax
import math
import random



def lax_scan_test():
    print("---- lax scan ----")
    print("1. Just find sum :)\n")
    
    def add(x,y):
        return x + y, x + y
    
    k = jax.random.key(42)
    arr = jax.random.randint(k, shape=(4096), minval=16, maxval=8192)


    then = perf_counter_ns()
    _, result = lax.scan(add, 0, arr)
    print(f"result: {result}, time: {(perf_counter_ns() - then) / 1000000}")

    then = perf_counter_ns()
    result = jnp.sum(arr)
    print(f"result: {result}, time: {(perf_counter_ns() - then) / 1000000}")

def lax_map_test():
    print(" ---- lax map -----")
    print(" ---- 2. Compare perf of jax.map with python map\n")

    k = jax.random.key(42)
    arr = jax.random.randint(k, shape=(1000000), minval=16, maxval=8192)
    arr_py = [random.randint(16, 8192) for _ in range(1000000)]
    

    def f_py(x):
        # just some ranom ops
        return math.atan(x) + math.pow(x, 7) - math.sqrt(x)
    
    
    def f_lax(x):
        # just some ranom ops
        return lax.atan(x) + lax.pow(x, 7) - lax.sqrt(x)
    
    then = perf_counter_ns()
    result = list(map(f_py, arr_py))
    print(f"result: {result[:4]}, time: {(perf_counter_ns() - then) / 1000000}")
    
    arr_float = jnp.astype(arr, jnp.float32)
    then = perf_counter_ns()
    result = lax.map(f_lax, arr_float) # it's slower for small arrays than python because of compilation and tracing overheads :)
    print(f"result: {result[:4]}, time: {(perf_counter_ns() - then) / 1000000}")

def lax_fori_test():
    print("---- lax fori ----")
    print("3. Just calculate max :)\n")

    max_val = 10000000

    def fun(idx, item):
        # just multiply by index
        return item * idx
    
    then = perf_counter_ns()
    result = lax.fori_loop(0, max_val, fun, 0)
    print(f"result: {result}, time: {(perf_counter_ns() - then) / 1000000}")

    # equivalent fori impl: https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop
    def fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val

    then = perf_counter_ns()
    result = fori_loop(0, max_val, fun, 0)
    print(f"result: {result}, time: {(perf_counter_ns() - then) / 1000000}")


def control_flow():
    lax_scan_test()
    lax_map_test()
    lax_fori_test()
    

if __name__ == "__main__":
    control_flow()