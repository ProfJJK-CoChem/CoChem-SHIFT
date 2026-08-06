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
        Strictly limits N to prevent exponential RAM exhaustion.
        """
        self.workspace = Path(workspace_path)
        self.n = n_spins
        
        # Guard against OOM (2^13 = 8192 x 8192 matrices = ~1.6GB per operator)
        if self.n > 12:
            raise MemoryError(f"❌ Spin system too large (N={self.n}). Direct diagonalization limit is 12. "
                              "Enable the Spin-Decoupling Override in Stage 1 to fragment the system.")
            
        print(f"⚙️ Initializing JAX Spin-1/2 Operators for N={self.n}...")
        self._build_operators()
        self.topological_mask = self._build_topological_mask(connectivity_matrix)

    def _build_operators(self):
        """Builds multi-spin Pauli matrices using Kronecker tensor products."""
        # Spin-1/2 fundamental Pauli matrices
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

        # Transfer cached arrays directly to GPU/Accelerator via JAX
        self.j_Ix = jax.device_put(self.Ix)
        self.j_Iy = jax.device_put(self.Iy)
        self.j_Iz = jax.device_put(self.Iz)
        print("✅ Spin operators cached in JAX device memory.")

    def _build_topological_mask(self, connectivity: np.ndarray) -> jnp.ndarray:
        """Uses NetworkX to mask J-couplings separated by > 4 bonds."""
        if connectivity is None:
            return jnp.ones((self.n, self.n))
            
        G = nx.from_numpy_array(connectivity)
        mask = np.zeros((self.n, self.n))
        
        # Calculate shortest path lengths
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
        H = Sum(v_i * Iz_i) + Sum(J_ij * (Ix_i*Ix_j + Iy_i*Iy_j + Iz_i*Iz_j))
        """
        # 1. Zeeman Term (Chemical Shifts in Hz)
        H_zeeman = jnp.einsum('i,ijk->jk', shifts, self.j_Iz)

        # 2. Apply Topological Mask to J-matrix
        j_masked = j_matrix * self.topological_mask
        
        # 3. J-Coupling Term (Scalar product of spin vectors)
        # We only sum i < j to avoid double counting
        H_j = jnp.zeros_like(H_zeeman)
        
        # Note: In a pure JAX context, static loops over N are unwrolled during JIT.
        # This is safe because N is strictly capped at 12.
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dot_product = (jnp.dot(self.j_Ix[i], self.j_Ix[j]) +
                               jnp.dot(self.j_Iy[i], self.j_Iy[j]) +
                               jnp.dot(self.j_Iz[i], self.j_Iz[j]))
                H_j += j_masked[i, j] * dot_product

        H_total = H_zeeman + H_j
        
        # 4. Diagonalize (Hermitian)
        eigenvalues, eigenvectors = jnp.linalg.eigh(H_total)
        return eigenvalues, eigenvectors

# Example initialization:
# engine = JAXSpinHamiltonian("./SHIFT_Workspace", n_spins=4)
# E, V = engine.solve_hamiltonian(jnp.array([100., 200., 300., 400.]), jnp.zeros((4,4)))