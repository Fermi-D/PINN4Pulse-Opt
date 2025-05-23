from flax import nnx
import jax
import jax.numpy as jnp

class Linear(nnx.Module):
  def __init__(self, num_input: int, num_output: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (num_input, num_output)))
    self.b = nnx.Param(jnp.zeros((num_output,)))
    self.num_input, self.num_output = num_input, num_output

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b

class MLP(nnx.Module):
    def __init__(self, num_input: int, num_output: int, rngs: nnx.Rngs):
        self.input_layer = Linear(num_input, 32, rngs=rngs)
        self.hidden_layers1 = Linear(32, 32, rngs=rngs)
        self.hidden_layers2 = Linear(32, 32, rngs=rngs)
        self.output_layer = Linear(32, num_output, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.tanh(self.input_layer(x))
        x = nnx.tanh(self.hidden_layers1(x))
        x = nnx.tanh(self.hidden_layers2(x))
        return self.output_layer(x)