import jax.numpy as jnp

def identity(dim: int) -> jax.Array:
    '''
    Returns the identity operator.

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns: 
      jnp.array: A 2x2 matrix of the Pauli X operator with complex64 precision.
    '''
    return jnp.identity(dim, dtype=jnp.complex64)

def pauli_x() -> jax.Array:
    '''
    Returns the Pauli X operator.
    The Pauli X operator is a 2x2 matrix defined as:
    [[0, 1],
     [1, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the Pauli X operator with complex64 precision.
    '''
    return jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)

def pauli_y() -> jax.Array:
    '''
    Returns the Pauli Y operator.
    The Pauli Y operator is a 2x2 matrix defined as:
    [[0, -i],
     [i, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the Pauli Y operator with complex64 precision.
    '''
    return jnp.array([[0+0j, 0-1j], [0+1j, 0+0j]], dtype=jnp.complex64)

def pauli_z() ->jax.Array:
    '''
    Returns the Pauli Z operator.
    The Pauli Z operator is a 2x2 matrix defined as:
    [[1, 0],
     [0, -1]]
    
    Returns:
      jnp.array: A 2x2 matrix of the Pauli Z operator with complex64 precision.
    '''
    return jnp.array([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=jnp.complex64)

def raising() -> jax.Array:
    '''
    Returns the raising operator.
    The raising operator is a 2x2 matrix defined as:
    [[0, 1],
     [0, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the raising operator with complex64 precision.
    '''
    return jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex64)

def lowering() -> jax.Array: 
    '''
    Returns the lowering operator.
    The lowering operator is a 2x2 matrix defined as:
    [[0, 0],
     [1, 0]]
    
    Returns:
      jnp.array: A 2x2 matrix of the lowering operator with complex64 precision.
    '''
    return jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.complex64)

def annihilate(dim: int) -> jax.Array:
    '''
    Returns the lowering operator.

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns:
      jnp.array: A 2x2 matrix of the annihilation operator with complex64 precision.
    '''
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=1)

def create(dim: int) -> jax.Array:
    '''
    Returns the lowering operator.

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns:
      jnp.array: A 'dim'x'dim' matrix of the creation operator with complex64 precision.
    '''
    a = annihilate(dim)
    return jax.lax.transpose(jax.lax.conj(a), (1, 0))

def num(dim: int) -> jax.Array:
    '''
    Returns the number operator.
    
    Args:
      dim (int): The dimension of Hilbert space.

    Returns:
      jnp.array: A 'dim'x'dim' matrix of the number operator with complex64 precision.
    '''
    return jnp.diag(jnp.arange(0, dim, dtype=jnp.complex64))