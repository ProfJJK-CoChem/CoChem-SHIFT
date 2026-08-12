from typing import Any, Dict, List, Optional
"""
Unit tests for CoChem-SHIFT Stage 4.1 JAX Spin Hamiltonian Engine.
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from cochem_shift_hamiltonian import JAXSpinHamiltonian, compute_deuterium_nqcc

def test_hamiltonian_construction_and_solve() -> None:
    n_spins = 3
    engine = JAXSpinHamiltonian(".", n_spins=n_spins)
    
    shifts = jnp.array([100.0, 200.0, 300.0])
    j_matrix = jnp.zeros((n_spins, n_spins))
    
    evals, evecs = engine.solve_hamiltonian(shifts, j_matrix)
    
    # 2^N eigenvalues = 8
    assert len(evals) == 2**n_spins
    assert not np.isnan(np.array(evals)).any()

def test_hamiltonian_oom_guard() -> None:
    with pytest.raises(MemoryError):
        JAXSpinHamiltonian(".", n_spins=15)

def test_2d_cosy_and_hsqc() -> None:
    n_spins = 2
    engine = JAXSpinHamiltonian(".", n_spins=n_spins)
    shifts = jnp.array([1.5, 7.2])
    j_matrix = jnp.array([[0.0, 7.5], [7.5, 0.0]])

    freqs, intensities = engine.compute_transition_intensities(shifts, j_matrix)
    assert len(freqs) > 0 and len(intensities) > 0

    x_grid, y_grid, cosy_mat = engine.generate_2d_cosy_matrix(np.array([1.5, 7.2]), np.array([[0.0, 7.5], [7.5, 0.0]]))
    assert cosy_mat.shape == (256, 256)
    assert np.max(cosy_mat) > 0.0

    h_grid, c_grid, hsqc_mat = engine.generate_2d_hsqc_matrix(np.array([1.5, 7.2]), np.array([25.0, 128.0]))
    assert hsqc_mat.shape == (256, 256)
    assert np.max(hsqc_mat) > 0.0

def test_deuterium_nqcc_basis_guard() -> None:
    efg = np.diag([1.0, -0.5, -0.5])
    # Quadruple-zeta core-valence basis sets pass
    res1 = compute_deuterium_nqcc(efg, basis_set="pcSseg-3")
    assert res1["basis_set_nqcc"] == "pcSseg-3"
    assert res1["provenance_tag"] == "[D]"
    assert res1["deuterium_nqcc_khz"] > 0

    res2 = compute_deuterium_nqcc(efg, basis_set="cc-pCVQZ")
    assert res2["basis_set_nqcc"] == "cc-pCVQZ"

    # Lower basis set fails with ValueError (§6.3)
    with pytest.raises(ValueError, match="strictly required"):
        compute_deuterium_nqcc(efg, basis_set="def2-SVP")