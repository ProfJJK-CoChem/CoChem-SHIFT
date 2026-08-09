"""
CoChem-SHIFT: Stage 4.1 - JAX Hamiltonian Construction & Caching
Filename: cochem_shift_hamiltonian.py

This module constructs the full quantum mechanical spin Hamiltonian for NMR.
It utilizes JAX for hardware-accelerated matrix diagonalization and NetworkX
for topological coupling truncation. The class pre-computes and caches 
spin operators in RAM to allow microsecond evaluations during MCMC fitting.
"""

import json
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
from functools import partial
from pathlib import Path

# Enable 64-bit precision in JAX for accurate matrix diagonalization
jax.config.update("jax_enable_x64", True)

class JAXSpinHamiltonian:
    def __init__(self, workspace_path: str, n_spins: int, connectivity_matrix: np.ndarray = None):
        """
        Initializes the spin operators for a given number of spin-1/2 nuclei.
        SHIFT-08: Clear spin limit explanation and weak-coupling options.
        SHIFT-09: Pre-computes two-body spin product operators I_dot[i, j] to avoid O(N^2) JIT loop dot products.
        """
        self.workspace = Path(workspace_path)
        self.n = n_spins
        
        # SHIFT-08: Direct diagonalization limit explanation
        if self.n > 12:
            raise MemoryError(f"Spin system size N={self.n} exceeds direct Hilbert space limit (2^{self.n} = {2**self.n}). "
                              "For N > 12, use weak-coupling block-diagonalization sub-spin partitioning.")
            
        print(f"⚙️ Initializing JAX Spin-1/2 Operators for N={self.n}...")
        self._build_operators()
        self.topological_mask = self._build_topological_mask(connectivity_matrix)

    def _build_operators(self):
        """
        Builds multi-spin Pauli matrices using Kronecker tensor products.
        SHIFT-09: Pre-computes two-body dot product operators I_xi I_xj + I_yi I_yj + I_zi I_zj.
        """
        I = np.eye(2, dtype=np.complex128)
        Sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) * 0.5
        Sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) * 0.5
        Sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) * 0.5

        dim = 2 ** self.n
        self.Ix = np.zeros((self.n, dim, dim), dtype=np.complex128)
        self.Iy = np.zeros((self.n, dim, dim), dtype=np.complex128)
        self.Iz = np.zeros((self.n, dim, dim), dtype=np.complex128)

        # Expand to full Hilbert space
        for i in range(self.n):
            op_x, op_y, op_z = 1.0, 1.0, 1.0
            for j in range(self.n):
                op_x = np.kron(op_x, Sx if i == j else I)
                op_y = np.kron(op_y, Sy if i == j else I)
                op_z = np.kron(op_z, Sz if i == j else I)
            
            self.Ix[i] = op_x
            self.Iy[i] = op_y
            self.Iz[i] = op_z

        # SHIFT-09: Pre-compute two-body spin product operators I_dot[i, j]
        self.I_dot = np.zeros((self.n, self.n, dim, dim), dtype=np.complex128)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dot_op = self.Ix[i] @ self.Ix[j] + self.Iy[i] @ self.Iy[j] + self.Iz[i] @ self.Iz[j]
                self.I_dot[i, j] = dot_op
                self.I_dot[j, i] = dot_op

        # Transfer cached arrays directly to GPU/Accelerator via JAX
        self.j_Ix = jax.device_put(self.Ix)
        self.j_Iy = jax.device_put(self.Iy)
        self.j_Iz = jax.device_put(self.Iz)
        self.j_I_dot = jax.device_put(self.I_dot)
        print("✅ Spin operators and pre-computed two-body dot product tensors cached in JAX memory.")

    def _build_topological_mask(self, connectivity: np.ndarray) -> jnp.ndarray:
        """Uses NetworkX to mask J-couplings separated by > 4 bonds."""
        if connectivity is None:
            return jnp.ones((self.n, self.n))
            
        G = nx.from_numpy_array(connectivity)
        mask = np.zeros((self.n, self.n))
        
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        for i in range(self.n):
            for j in range(self.n):
                if lengths.get(i, {}).get(j, 999) <= 4:
                    mask[i, j] = 1.0
                    
        print("🕸️ Topological Cutoff Mask generated (Max 4 bonds).")
        return jax.device_put(mask)

    @partial(jax.jit, static_argnums=(0,))
    def solve_hamiltonian(self, shifts: jnp.ndarray, j_matrix: jnp.ndarray):
        """
        JIT-compiled strong-coupling Hamiltonian assembly and diagonalization.
        SHIFT-09: Fast pre-computed two-body tensor lookup.
        """
        # 1. Zeeman Term (Chemical Shifts in Hz)
        H_zeeman = jnp.einsum('i,ijk->jk', shifts, self.j_Iz)

        # 2. Apply Topological Mask to J-matrix
        j_masked = j_matrix * self.topological_mask
        
        # 3. SHIFT-09: Fast J-Coupling Term using pre-computed j_I_dot
        H_j = jnp.einsum('ij,ijkl->kl', j_masked * 0.5, self.j_I_dot)

        H_total = H_zeeman + H_j
        
        # 4. Diagonalize (Hermitian)
        eigenvalues, eigenvectors = jnp.linalg.eigh(H_total)
        return eigenvalues, eigenvectors