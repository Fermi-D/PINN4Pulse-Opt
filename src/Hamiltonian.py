import jax
import jax.numpy as jnp

import hydra
from omegaconf import DictConfig

@hydra.main(config_name="system_params", version_base=None, config_path="./config")
def control(t: float, cfg: DictConfig):
    '''
    Control Hamiltonian for the resonator-coupled transmon in Eq.(7)
    
    Args:
      t (float): Current time (in seconds) at which to evaluate the control Hamiltonian.
      cfg (DictConfig): Hydra‐loaded configuration object containing system parameters, 
        for example:
        - cfg.transmon.dim: dimension of the Hilbert space  
        - cfg.transmon.anharmonicity: anharmonicity of the transmon  
        - any other nested fields defining pulse characteristics or coupling strengths
      
    Returns:
      jnp.array: Matrix representation of the control Hamiltonian (see system_params.yaml for matrix size)
    '''
    I = qo.identity(cfg.transmon.dim)
    num = qo.num(cfg.transmon.dim)
    Omega, delta = pulse(t, model)
    
    return delta*num + (cfg.transmon.anharmonicity/2)*jnp.dot(num, num-I) + (Omega*a+jax.lax.conj(Omega)*a_dag)/2

@hydra.main(config_name="system_params", version_base=None, config_path="./config")
def error(t: float, cfg: DictConfig):
    '''
    Error Hamiltonian (delta(t)->delta(t)+epsilon) corresponding to Eq.(7)

    Args:
      t (float): Current time (in seconds) at which to evaluate the control Hamiltonian.
      cfg (DictConfig): Hydra‐loaded configuration object containing system parameters, 
        for example:
        - cfg.transmon.dim: dimension of the Hilbert space  
        - cfg.transmon.anharmonicity: anharmonicity of the transmon  
        - any other nested fields defining pulse characteristics or coupling strengths

    Returns:
      jnp.array: Matrix representation of the error Hamiltonian (see system_params.yaml for matrix size)
    '''
    return cfg.error.strength * qo.num(cfg.transmon.dim)