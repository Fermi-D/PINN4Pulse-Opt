import jax
import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock # For mocking Hydra's DictConfig

@hydra.main(config_name="system_params", version_base=None, config_path="./config")
def smoothed_square(t: float, cfg: DictConfig):
    '''
    Smoothed unit square pulse expressed as follows:
      A(t) = coth(kappa*T){tanh(kappa*T) - tanh(kappa(t-T))} - 1
    
    Args:
      t (float): Current time (in seconds) at which to evaluate the control Hamiltonian.
      cfg (DictConfig): Hydra‐loaded configuration object containing system parameters, 
        for example:
        - cfg.drive.kappa: the degree of the smoothing
        - cfg.drive.duration: anharmonicity of the transmon  
      
    Returns:
      jnp.array: Matrix representation of the envelope (see system_params.yaml for kappa and duration)
    '''
    term_1 = (1/jax.lax.tanh(cfg.drive.kappa*cfg.drive.duration)) * jax.lax.tanh(cfg.drive.kappa*t)
    term_2 = (1/jax.lax.tanh(cfg.drive.kappa*cfg.drive.duration)) * jax.lax.tanh(cfg.drive.kappa*(t-cfg.drive.duration))
    return term_1 - term_2 - 1

class TestSmoothedSquare:

    @pytest.fixture
    def mock_cfg(self):
        """
        Fixture to provide a mock DictConfig object for testing.
        Simulates the structure of your system_params.yaml.
        """
        # Create a mock object that behaves like a DictConfig
        cfg = MagicMock()
        cfg.drive.kappa = 5.0  # Example value for kappa
        cfg.drive.duration = 10.0 # Example value for duration
        return cfg

    def test_smoothed_square_at_start(self, mock_cfg: MagicMock):
        """
        Tests the function's value at t=0 (start of the pulse).
        A well-behaved smoothed square pulse should be close to 0 at t=0.
        """
        t = 0.0
        result = smoothed_square(t, mock_cfg)
        
        # Expected value at t=0:
        # A(0) = coth(k*T){tanh(0) - tanh(-k*T)} - 1
        #      = coth(k*T){0 - (-tanh(k*T))} - 1
        #      = coth(k*T) * tanh(k*T) - 1
        #      = 1 - 1 = 0
        expected = 0.0
        assert jnp.allclose(result, expected, atol=1e-6) # Use atol for absolute tolerance

    def test_smoothed_square_at_duration(self, mock_cfg: MagicMock):
        """
        Tests the function's value at t=duration (end of the pulse).
        A well-behaved smoothed square pulse should be close to 0 at t=duration.
        """
        t = mock_cfg.drive.duration
        result = smoothed_square(t, mock_cfg)
        
        # Expected value at t=T (duration):
        # A(T) = coth(k*T){tanh(k*T) - tanh(k*(T-T))} - 1
        #      = coth(k*T){tanh(k*T) - tanh(0)} - 1
        #      = coth(k*T){tanh(k*T) - 0} - 1
        #      = coth(k*T) * tanh(k*T) - 1
        #      = 1 - 1 = 0
        expected = 0.0
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_smoothed_square_at_midpoint(self, mock_cfg: MagicMock):
        """
        Tests the function's value at t=duration/2 (midpoint of the pulse).
        For an ideal unit square pulse, this would be 1.
        """
        t = mock_cfg.drive.duration / 2.0
        result = smoothed_square(t, mock_cfg)

        # Analytical value at t=T/2:
        # A(T/2) = coth(k*T){tanh(k*T/2) - tanh(k*(-T/2))} - 1
        #        = coth(k*T){tanh(k*T/2) + tanh(k*T/2)} - 1
        #        = coth(k*T) * 2 * tanh(k*T/2) - 1
        
        expected_midpoint = (1/jnp.tanh(mock_cfg.drive.kappa*mock_cfg.drive.duration)) * 2 * jnp.tanh(mock_cfg.drive.kappa*(mock_cfg.drive.duration/2)) - 1
        assert jnp.allclose(result, expected_midpoint)
        # For typical kappa values, the smoothed pulse should be close to 1 at the midpoint
        assert result > 0.9 

    def test_smoothed_square_with_different_params(self):
        """
        Tests the function with different kappa (smoothing) and duration values.
        """
        cfg = MagicMock()
        cfg.drive.kappa = 20.0 # Sharper pulse (less smoothing)
        cfg.drive.duration = 5.0 # Shorter duration

        t = 0.0
        result_start = smoothed_square(t, cfg)
        assert jnp.allclose(result_start, 0.0, atol=1e-6)

        t = cfg.drive.duration
        result_end = smoothed_square(t, cfg)
        assert jnp.allclose(result_end, 0.0, atol=1e-6)

        t = cfg.drive.duration / 2.0
        result_mid = smoothed_square(t, cfg)
        expected_midpoint = (1/jnp.tanh(cfg.drive.kappa*cfg.drive.duration)) * 2 * jnp.tanh(cfg.drive.kappa*(cfg.drive.duration/2)) - 1
        assert jnp.allclose(result_mid, expected_midpoint)
        # With a larger kappa, it should be even closer to 1
        assert result_mid > 0.99 

    def test_smoothed_square_with_jax_jit(self, mock_cfg: MagicMock):
        """
        Tests if the function is JIT-compilable, a key feature for JAX performance.
        """
        # JIT compile the function
        jit_smoothed_square = jax.jit(smoothed_square)

        t = 1.0
        # Call the JIT-compiled function. If it compiles without error, the test passes.
        result = jit_smoothed_square(t, mock_cfg)
        
        # Verify the result against the non-JIT compiled version
        expected_result = smoothed_square(t, mock_cfg)
        assert jnp.allclose(result, expected_result)
        # Also check the type to ensure it's a JAX Array
        assert isinstance(result, jax.Array)

    def test_smoothed_square_grad(self, mock_cfg: MagicMock):
        """
        Tests if the function can be automatically differentiated by JAX.
        This is crucial for optimization and control problems.
        """
        # Define a wrapper function to differentiate with respect to 't_val'
        def func_to_differentiate(t_val):
            return smoothed_square(t_val, mock_cfg)

        # Compute the gradient
        grad_fn = jax.grad(func_to_differentiate)

        t_test = mock_cfg.drive.duration / 4.0 # Pick a point within the pulse
        gradient_at_t = grad_fn(t_test)

        # Gradients should be finite and generally non-zero at most points in the pulse
        assert jnp.isfinite(gradient_at_t)
        assert gradient_at_t != 0.0 # Expect a non-zero gradient here