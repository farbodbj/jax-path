# Training a simpel neural network with jax
import jax
import jax.numpy as jnp


@jax.jit
def mse_loss(y: jnp.ndarray, y_hat: jnp.ndarray):
    assert y.shape == y_hat.shape
    return  jnp.sum((y - y_hat) ** 2) / y.shape[0]

@jax.jit
def linear_layer(W: jnp.ndarray, x:jnp.ndarray, b: jnp.ndarray)->jnp.ndarray:
    return x @ W + b

@jax.jit
def relu(X: jnp.ndarray):
    return jnp.maximum(X, jnp.zeros_like(X))
    

@jax.jit
def forward(params, X):
    W_1, b_1, W_2, b_2 = params
    
    y_1 = relu(linear_layer(W_1, X, b_1))
    return relu(linear_layer(W_2, y_1, b_2))

@jax.jit
def loss_fn(params, X, Y):
    y_hat = forward(params, X)
    return mse_loss(Y, y_hat)


if __name__ == "__main__":
    key = jax.random.key(42)
    k1,k2,k3,k4,k5,k6 = jax.random.split(key, 6)
    NUM_FEATURES = 256
    EPOCH = 4
    LR = 1e-4

    W_1 = jax.random.normal(k1, shape=(NUM_FEATURES, 64))
    b_1 = jax.random.normal(k2, shape=(1, 64))

    W_2 = jax.random.normal(k3, shape=(64, 1))
    b_2 = jax.random.normal(k4, shape=(1, 1))

    X = jax.random.normal(k5, shape=(1024, NUM_FEATURES))
    Y = jax.random.normal(k6, shape=(1024, 1))
    
    for _ in range(EPOCH):
        params = (W_1, b_1, W_2, b_2)
        loss_value, (dW1, db1, dW2, db2) = jax.value_and_grad(loss_fn)(params, X, Y)
        
        W_1 -= LR * dW1
        b_1 -= LR * db1
        W_2 -= LR * dW2
        b_2 -= LR * db2

        print(f"Loss: {loss_value}")




