import jax
import jax.numpy as jnp

from flax import nnx
import optax

import hydra
from omegaconf import DictConfig

def loss(