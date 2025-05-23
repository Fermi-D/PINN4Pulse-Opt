import jax
import jax.numpy as jnp
from flax import nnx
import pytest

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

class TestLinear:
    @pytest.fixture
    def rngs(self):
        """Provides a fresh Rngs instance for each test."""
        # Use different seeds for params and dropout to ensure isolation
        return nnx.Rngs(params=0, dropout=1)

    def test_linear_init(self, rngs: nnx.Rngs):
        """Tests the initialization of the Linear layer."""
        num_input = 10
        num_output = 5
        model = Linear(num_input, num_output, rngs=rngs)

        # Verify the existence, type, and shape of parameters (w and b)
        assert isinstance(model.w, nnx.Param)
        assert model.w.value.shape == (num_input, num_output)
        # Default dtype for jax.random.uniform and jnp.zeros is float32
        assert model.w.value.dtype == jnp.float32

        assert isinstance(model.b, nnx.Param)
        assert model.b.value.shape == (num_output,)
        assert model.b.value.dtype == jnp.float32

        # Verify stored input/output dimensions
        assert model.num_input == num_input
        assert model.num_output == num_output

    def test_linear_call(self, rngs: nnx.Rngs):
        """Tests the forward pass (__call__) of the Linear layer."""
        num_input = 10
        num_output = 5
        model = Linear(num_input, num_output, rngs=rngs)

        batch_size = 4
        # Create a dummy input array
        x = jnp.ones((batch_size, num_input), dtype=jnp.float32)

        output = model(x)

        # Verify output shape and dtype
        assert output.shape == (batch_size, num_output)
        assert output.dtype == jnp.float32

        # Basic verification of the computation:
        # If input 'x' is all ones, then x @ w results in a row vector
        # where each element is the sum of the corresponding column in w.
        # Then, 'b' is added to this vector.
        expected_output_row = jnp.sum(model.w.value, axis=0) + model.b.value
        # All rows in the output should be identical for this specific input
        assert jnp.allclose(output, jnp.tile(expected_output_row, (batch_size, 1)))

    def test_linear_rng_reproducibility(self):
        """Ensures that Linear layer initialization is reproducible with the same RNG seed."""
        num_input = 10
        num_output = 5

        # Initialize two models with the same seed
        rngs1 = nnx.Rngs(params=42)
        model1 = Linear(num_input, num_output, rngs=rngs1)

        rngs2 = nnx.Rngs(params=42)
        model2 = Linear(num_input, num_output, rngs=rngs2)

        # Verify that their parameters are identical
        assert jnp.allclose(model1.w.value, model2.w.value)
        assert jnp.allclose(model1.b.value, model2.b.value)

    def test_linear_grad(self, rngs: nnx.Rngs):
        """Tests that gradients can be computed for the Linear layer."""
        num_input = 10
        num_output = 5
        model = Linear(num_input, num_output, rngs=rngs)

        batch_size = 4
        x = jnp.ones((batch_size, num_input), dtype=jnp.float32)

        # Define a dummy loss function (e.g., mean squared error)
        def loss_fn(params, model_args, inputs, targets):
            # Replace model parameters for gradient calculation with nnx.grad
            model_ref = model.replace(params)
            outputs = model_ref(*model_args, inputs)
            return jnp.mean((outputs - targets)**2)

        # Dummy target values
        targets = jnp.zeros((batch_size, num_output), dtype=jnp.float32)

        # Compute gradients with respect to "params"
        grads = nnx.grad(loss_fn, "params")(model.params(), (x,), x, targets)

        # Verify that gradients exist for 'w' and 'b' and have the correct shape
        assert "w" in grads
        assert "b" in grads
        assert grads["w"].shape == (num_input, num_output)
        assert grads["b"].shape == (num_output,)
        
        # Ensure gradients are not all zeros (highly unlikely for random init)
        assert not jnp.allclose(grads["w"], jnp.zeros_like(grads["w"]))
        assert not jnp.allclose(grads["b"], jnp.zeros_like(grads["b"]))


class TestMLP:
    @pytest.fixture
    def rngs(self):
        """Provides a fresh Rngs instance for each test."""
        return nnx.Rngs(params=0, dropout=1)

    def test_mlp_init(self, rngs: nnx.Rngs):
        """Tests the initialization of the MLP model."""
        num_input = 10
        num_output = 2
        model = MLP(num_input, num_output, rngs=rngs)

        # Verify that all Linear layers are correctly instantiated
        assert isinstance(model.input_layer, Linear)
        assert isinstance(model.hidden_layers1, Linear)
        assert isinstance(model.hidden_layers2, Linear)
        assert isinstance(model.output_layer, Linear)

        # Verify that parameters for all layers exist within the model's parameters
        params = model.params()
        assert "input_layer" in params
        assert "hidden_layers1" in params
        assert "hidden_layers2" in params
        assert "output_layer" in params

        # Verify shapes of some key parameters
        assert params["input_layer"]["w"].shape == (num_input, 32)
        assert params["output_layer"]["w"].shape == (32, num_output)

    def test_mlp_call(self, rngs: nnx.Rngs):
        """Tests the forward pass (__call__) of the MLP model."""
        num_input = 10
        num_output = 2
        model = MLP(num_input, num_output, rngs=rngs)

        batch_size = 4
        x = jnp.ones((batch_size, num_input), dtype=jnp.float32)

        output = model(x)

        # Verify output shape and dtype
        assert output.shape == (batch_size, num_output)
        assert output.dtype == jnp.float32

    def test_mlp_rng_reproducibility(self):
        """Ensures that MLP model initialization is reproducible with the same RNG seed."""
        num_input = 10
        num_output = 2

        # Initialize two models with the same seed
        rngs1 = nnx.Rngs(params=100)
        model1 = MLP(num_input, num_output, rngs=rngs1)

        rngs2 = nnx.Rngs(params=100)
        model2 = MLP(num_input, num_output, rngs=rngs2)

        # Recursively compare all parameters of the two models
        def compare_params(p1, p2):
            if isinstance(p1, dict) and isinstance(p2, dict):
                assert set(p1.keys()) == set(p2.keys())
                for k in p1:
                    compare_params(p1[k], p2[k])
            elif isinstance(p1, jax.Array) and isinstance(p2, jax.Array):
                assert jnp.allclose(p1, p2)
            elif isinstance(p1, nnx.Param) and isinstance(p2, nnx.Param):
                assert jnp.allclose(p1.value, p2.value)
            else:
                assert p1 == p2 # Compare other attributes if any (for robustness)

        compare_params(model1.params(), model2.params())


    def test_mlp_grad(self, rngs: nnx.Rngs):
        """Tests that gradients can be computed for the MLP model."""
        num_input = 10
        num_output = 2
        model = MLP(num_input, num_output, rngs=rngs)

        batch_size = 4
        x = jnp.ones((batch_size, num_input), dtype=jnp.float32)

        def loss_fn(params, model_args, inputs, targets):
            model_ref = model.replace(params)
            outputs = model_ref(*model_args, inputs)
            return jnp.mean((outputs - targets)**2)

        targets = jnp.zeros((batch_size, num_output), dtype=jnp.float32)

        # Compute gradients for all model parameters
        grads = nnx.grad(loss_fn, "params")(model.params(), (x,), x, targets)

        # Verify that gradients exist for each layer
        assert "input_layer" in grads
        assert "hidden_layers1" in grads
        assert "hidden_layers2" in grads
        assert "output_layer" in grads

        # Verify the shape of gradients for specific layers (input and output)
        assert grads["input_layer"]["w"].shape == (num_input, 32)
        assert grads["input_layer"]["b"].shape == (32,)
        assert grads["output_layer"]["w"].shape == (32, num_output)
        assert grads["output_layer"]["b"].shape == (num_output,)
        
        # Ensure gradients are not all zeros (highly unlikely for random init)
        assert not jnp.allclose(grads["input_layer"]["w"], jnp.zeros_like(grads["input_layer"]["w"]))
        assert not jnp.allclose(grads["output_layer"]["b"], jnp.zeros_like(grads["output_layer"]["b"]))