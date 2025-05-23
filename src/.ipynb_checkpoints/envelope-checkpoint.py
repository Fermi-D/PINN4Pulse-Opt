import jax
import jax.numpy as jnp

import hydra
from omegaconf import DictConfig

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