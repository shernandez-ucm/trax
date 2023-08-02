import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
from optax import rmsprop


def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    #return series[..., np.newaxis].astype(np.float32)
    return series

class LSTMModel(nn.Module):
        
    @nn.compact   
    def __call__(self, X_batch):
        x=nn.RNN(nn.LSTMCell(64))(X_batch)
        x=nn.Dense(10)(x)
        return x


np.random.seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]


batch_size = 32

key=jax.random.PRNGKey(32)
inputs = jax.random.randint(key,(batch_size, n_steps),0, 10,)
#inputs=inputs[:,:,np.newaxis].astype(jnp.float32)

#for epoch in range(10):
#    loss, state = train_state.minimize(model, inputs, state, opt)
#    print(loss)
