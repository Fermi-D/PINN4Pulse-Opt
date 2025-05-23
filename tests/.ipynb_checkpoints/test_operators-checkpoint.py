import jax
import jax.numpy as jnp
import pytest
import numpy as np

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
    Returns the annihilation operator.

    Args:
      dim (int): The dimension of Hilbert space.
      
    Returns:
      jnp.array: A 'dim'x'dim' matrix of the annihilation operator with complex64 precision.
    '''
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim, dtype=jnp.complex64)), k=1)

def create(dim: int) -> jax.Array:
    '''
    Returns the creation operator.

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

class TestOperators:

    def test_identity(self):
        id2 = identity(2)
        assert id2.shape == (2, 2)
        assert id2.dtype == jnp.complex64
        expected = jnp.array([[1+0j, 0+0j], [0+0j, 1+0j]], dtype=jnp.complex64)
        assert jnp.allclose(id2, expected)

        id3 = identity(3)
        assert id3.shape == (3, 3)
        assert id3.dtype == jnp.complex64
        expected = jnp.array([[1+0j, 0+0j, 0+0j],
                              [0+0j, 1+0j, 0+0j],
                              [0+0j, 0+0j, 1+0j]], dtype=jnp.complex64)
        assert jnp.allclose(id3, expected)

    def test_pauli_x(self):
        px = pauli_x()
        assert px.shape == (2, 2)
        assert px.dtype == jnp.complex64
        expected = jnp.array([[0+0j, 1+0j], [1+0j, 0+0j]], dtype=jnp.complex64)
        assert jnp.allclose(px, expected)

    def test_pauli_y(self):
        py = pauli_y()
        assert py.shape == (2, 2)
        assert py.dtype == jnp.complex64
        # Pauli Yの物理的に正しい定義でテストします。
        expected = jnp.array([[0+0j, 0-1j], [0+1j, 0+0j]], dtype=jnp.complex64)
        assert jnp.allclose(py, expected)

    def test_pauli_z(self):
        pz = pauli_z()
        assert pz.shape == (2, 2)
        assert pz.dtype == jnp.complex64
        expected = jnp.array([[1+0j, 0+0j], [0+0j, -1+0j]], dtype=jnp.complex64)
        assert jnp.allclose(pz, expected)

    def test_raising(self):
        r_op = raising()
        assert r_op.shape == (2, 2)
        assert r_op.dtype == jnp.complex64
        expected = jnp.array([[0+0j, 1+0j], [0+0j, 0+0j]], dtype=jnp.complex64)
        assert jnp.allclose(r_op, expected)

    def test_lowering(self):
        l_op = lowering()
        assert l_op.shape == (2, 2)
        assert l_op.dtype == jnp.complex64
        expected = jnp.array([[0+0j, 0+0j], [1+0j, 0+0j]], dtype=jnp.complex64)
        assert jnp.allclose(l_op, expected)

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_annihilate(self, dim):
        a_op = annihilate(dim)
        assert a_op.shape == (dim, dim)
        assert a_op.dtype == jnp.complex64
        
        expected_diag_elements = np.sqrt(np.arange(1, dim), dtype=np.complex64)
        
        expected_matrix = jnp.zeros((dim, dim), dtype=jnp.complex64)
        for i in range(dim - 1):
            expected_matrix = expected_matrix.at[i, i+1].set(expected_diag_elements[i])
        
        assert jnp.allclose(a_op, expected_matrix)
        
        assert jnp.allclose(jnp.diag(a_op, k=1), expected_diag_elements)
        if dim > 1:
            assert jnp.allclose(jnp.diag(a_op, k=0), jnp.zeros(dim, dtype=jnp.complex64))
            assert jnp.allclose(jnp.diag(a_op, k=-1), jnp.zeros(dim-1, dtype=jnp.complex64))

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_create(self, dim):
        c_op = create(dim)
        assert c_op.shape == (dim, dim)
        assert c_op.dtype == jnp.complex64
        
        a_op = annihilate(dim)
        expected_c_op = jnp.transpose(jnp.conjugate(a_op))
        
        assert jnp.allclose(c_op, expected_c_op)

        expected_diag_elements = np.sqrt(np.arange(1, dim), dtype=np.complex64)
        assert jnp.allclose(jnp.diag(c_op, k=-1), expected_diag_elements)

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_num(self, dim):
        n_op = num(dim)
        assert n_op.shape == (dim, dim)
        assert n_op.dtype == jnp.complex64
        
        expected_diag_elements = np.arange(0, dim, dtype=jnp.complex64)
        
        expected_matrix = jnp.diag(expected_diag_elements)
        
        assert jnp.allclose(n_op, expected_matrix)
        
        assert jnp.allclose(jnp.diag(n_op), expected_diag_elements)
        assert jnp.allclose(n_op - jnp.diag(jnp.diag(n_op)), jnp.zeros_like(n_op))