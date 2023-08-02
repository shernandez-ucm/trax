import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from optax import rmsprop

from flax import linen as nn

class LSTMModel(nn.Module):
    def setup(self):
        LSTMLayer = nn.scan(nn.OptimizedLSTMCell,
                               variable_broadcast="params",
                               split_rngs={"params": False},
                               in_axes=1, out_axes=1,
                               length=10,
                               reverse=False)
        self.lstm = LSTMLayer(name="LSTM")
        self.linear1 = nn.Dense(1, name="Dense1")
        
    @nn.remat    
    def __call__(self, X_batch):
        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(X_batch),), size=20)
        x=X_batch
        (carry, hidden), x = self.lstm((carry, hidden), x)
        return self.linear1(x[:, -1])


hidden_size = 10
num_layers = 2
batch_size = 32
seq_len = 10

key=jax.random.PRNGKey(32)
inputs = jax.random.randint(key,(batch_size, seq_len),0, 10,)


#for epoch in range(10):
#    loss, state = train_state.minimize(model, inputs, state, opt)
#    print(loss)
