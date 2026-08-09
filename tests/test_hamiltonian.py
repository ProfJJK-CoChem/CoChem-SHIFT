"""
Unit tests for CoChem-SHIFT Stage 4.1 JAX Spin Hamiltonian Engine.
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from cochem_shift_hamiltonian import JAXSpinHamiltonian

def test_hamiltonian_construction_and_solve():
    n_spins = 3
    engine = JAXSpinHamiltonian(".", n_spins=n_spins)
    
    shifts = jnp.array([100.0, 200.0, 300.0])
    j_matrix = jnp.zeros((n_spins, n_spins))
    
    evals, evecs = engine.solve_hamiltonian(shifts, j_matrix)
    
    # 2^N eigenvalues = 8
    assert len(evals) == 2**n_spins
    assert not np.isnan(np.array(evals)).any()

def test_hamiltonian_oom_guard():
    with pytest.raises(MemoryError):
        JAXSpinHamiltonian(".", n_spins=15)
