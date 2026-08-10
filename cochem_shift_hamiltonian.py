"""
CoChem-SHIFT: Stage 4.1 - JAX Hamiltonian Construction & Caching
Filename: cochem_shift_hamiltonian.py

This module constructs the full quantum mechanical spin Hamiltonian for NMR.
It utilizes JAX for hardware-accelerated matrix diagonalization and NetworkX
for topological coupling truncation. The class pre-computes and caches 
spin operators in RAM to allow microsecond evaluations during MCMC fitting.
"""

import json
import math
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

    def compute_transition_intensities(self, shifts: jnp.ndarray, j_matrix: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes quantum mechanical transition frequencies nu_ab and dipole intensities I_ab:
        I_ab = |<psi_a | F_x | psi_b>|^2 where F_x = sum_i I_xi.
        """
        evals, evecs = self.solve_hamiltonian(shifts, j_matrix)
        evals_np = np.array(evals)
        evecs_np = np.array(evecs) # shape (dim, dim) where evecs_np[:, a] is psi_a

        # Total transverse spin operator F_x = sum_i Ix[i]
        Fx = np.sum(self.Ix, axis=0)

        dim = len(evals_np)
        freqs = []
        intensities = []

        for a in range(dim):
            psi_a = evecs_np[:, a]
            for b in range(a + 1, dim):
                psi_b = evecs_np[:, b]
                # Matrix element <psi_a | F_x | psi_b>
                trans_element = float(np.abs(psi_a.conj() @ Fx @ psi_b) ** 2)
                if trans_element > 1e-6:
                    nu_ab = abs(evals_np[a] - evals_np[b])
                    freqs.append(nu_ab)
                    intensities.append(trans_element)

        return np.array(freqs, dtype=float), np.array(intensities, dtype=float)

    def generate_2d_cosy_matrix(self, shifts_ppm: np.ndarray, j_matrix: np.ndarray, 
                                grid_size: int = 256, ppm_range: Tuple[float, float] = (0.0, 10.0), 
                                hwhm: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 2D homonuclear COSY (1H-1H) correlation matrix I(nu_1, nu_2).
        Includes diagonal peaks at (delta_i, delta_i) and off-diagonal cross-peaks at (delta_i, delta_j).
        """
        ppm_grid = np.linspace(ppm_range[0], ppm_range[1], grid_size)
        x_grid, y_grid = np.meshgrid(ppm_grid, ppm_grid)
        cosy_matrix = np.zeros((grid_size, grid_size), dtype=float)

        n = min(len(shifts_ppm), self.n)
        gamma = max(1e-4, hwhm)

        # 1. Diagonal Peaks
        for i in range(n):
            d_i = shifts_ppm[i]
            lorentzian_x = gamma / (math.pi * ((x_grid - d_i)**2 + gamma**2))
            lorentzian_y = gamma / (math.pi * ((y_grid - d_i)**2 + gamma**2))
            cosy_matrix += lorentzian_x * lorentzian_y * 2.0

        # 2. Cross Peaks
        for i in range(n):
            for j in range(i + 1, n):
                j_val = abs(j_matrix[i, j]) if j_matrix is not None and i < j_matrix.shape[0] and j < j_matrix.shape[1] else 0.0
                if j_val > 0.5 or n <= 4:
                    d_i, d_j = shifts_ppm[i], shifts_ppm[j]
                    amp = max(0.2, j_val / 10.0)
                    
                    # (d_i, d_j) and (d_j, d_i)
                    lor_1 = (gamma / (math.pi * ((x_grid - d_i)**2 + gamma**2))) * (gamma / (math.pi * ((y_grid - d_j)**2 + gamma**2)))
                    lor_2 = (gamma / (math.pi * ((x_grid - d_j)**2 + gamma**2))) * (gamma / (math.pi * ((y_grid - d_i)**2 + gamma**2)))
                    cosy_matrix += amp * (lor_1 + lor_2)

        return ppm_grid, ppm_grid, cosy_matrix

    def generate_2d_hsqc_matrix(self, h_shifts_ppm: np.ndarray, c_shifts_ppm: np.ndarray, 
                                bond_matrix: np.ndarray = None, grid_size: int = 256, 
                                h_range: Tuple[float, float] = (0.0, 10.0), 
                                c_range: Tuple[float, float] = (0.0, 200.0), 
                                hwhm_h: float = 0.02, hwhm_c: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 2D heteronuclear HSQC (1H-13C) correlation matrix I(nu_H, nu_C).
        Places single-bond cross-peaks at (delta_H_i, delta_C_k).
        """
        h_grid = np.linspace(h_range[0], h_range[1], grid_size)
        c_grid = np.linspace(c_range[0], c_range[1], grid_size)
        x_grid, y_grid = np.meshgrid(h_grid, c_grid)
        hsqc_matrix = np.zeros((grid_size, grid_size), dtype=float)

        gamma_h = max(1e-4, hwhm_h)
        gamma_c = max(1e-4, hwhm_c)

        n_h = len(h_shifts_ppm)
        n_c = len(c_shifts_ppm)

        for i in range(n_h):
            for k in range(n_c):
                # Check single-bond CH correlation
                is_bonded = True
                if bond_matrix is not None and i < bond_matrix.shape[0] and k < bond_matrix.shape[1]:
                    is_bonded = bool(bond_matrix[i, k])
                    
                if is_bonded:
                    dh = h_shifts_ppm[i]
                    dc = c_shifts_ppm[k]
                    lor_h = gamma_h / (math.pi * ((x_grid - dh)**2 + gamma_h**2))
                    lor_c = gamma_c / (math.pi * ((y_grid - dc)**2 + gamma_c**2))
                    hsqc_matrix += lor_h * lor_c

        return h_grid, c_grid, hsqc_matrix


def compute_deuterium_nqcc(efg_tensor: np.ndarray, basis_set: str = "pcSseg-3") -> Dict[str, Any]:
    """
    Enforces Core-Valence Quadruple-Zeta basis set (pcSseg-3 or cc-pCVQZ) for Deuterium NQCC (§6.3).
    Computes Nuclear Quadrupole Coupling Constant chi (kHz) from Electric Field Gradient (EFG) tensor.
    """
    valid_basis_sets = ["pcSseg-3", "cc-pCVQZ"]
    if basis_set not in valid_basis_sets:
        raise ValueError(
            f"Invalid basis set '{basis_set}' for Deuterium NQCC calculation. "
            f"Core-Valence Quadruple-Zeta basis set ({valid_basis_sets}) is strictly required (§6.3)."
        )

    efg_sym = 0.5 * (np.asarray(efg_tensor) + np.asarray(efg_tensor).T)
    evals = np.sort(np.abs(np.linalg.eigvalsh(efg_sym)))
    q_zz = float(evals[-1])  # largest principal component in atomic units

    conversion_factor_khz = 67.17105
    chi_khz = float(q_zz * conversion_factor_khz)

    return {
        "deuterium_nqcc_khz": chi_khz,
        "basis_set_nqcc": basis_set,
        "provenance_tag": "[D]"
    }