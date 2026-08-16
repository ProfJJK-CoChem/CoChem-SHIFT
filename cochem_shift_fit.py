import hashlib
import logging
logger = logging.getLogger(__name__)
"""
CoChem-SHIFT: Stage 4.2 - MCMC Bayesian Spin Fitting Engine
Filename: cochem_shift_fit.py

This module performs Bayesian optimization of nuclear chemical shifts and 
J-couplings using Markov Chain Monte Carlo (MCMC). It ingests theoretical
tensors as the prior expectation, utilizes the JAX-compiled Hamiltonian to 
rapidly simulate spectral states, and evaluates a likelihood function against 
the provided experimental JCAMP-DX peak data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# SHIFT-17: Proper JAX import handling with explicit error on execution
HAS_JAX = False
try:
    from cochem_shift_hamiltonian import JAXSpinHamiltonian
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError as err:
    import logging
    logging.getLogger(__name__).warning("JAX or JAXSpinHamiltonian not available (%s). Bayesian spin fitting requires JAX.", err)
    HAS_JAX = False

def apply_template_anchoring(
    calc_shieldings: np.ndarray,
    ref_shielding: float,
    parent_exp_shifts: Dict[str, float],
    parent_calc_shieldings: Dict[str, float]
) -> np.ndarray:
    """
    Applies Product Class B/C template anchoring (§1.2).
    delta_calc = sigma_ref - sigma_calc + Delta_template
    where Delta_template = mean(delta_exp_parent - (sigma_ref - sigma_calc_parent))
    """
    calc_shieldings = np.asarray(calc_shieldings)
    raw_shifts = ref_shielding - calc_shieldings
    if not parent_exp_shifts or not parent_calc_shieldings:
        return raw_shifts

    offsets = []
    for k, exp_val in parent_exp_shifts.items():
        if k in parent_calc_shieldings:
            parent_raw = ref_shielding - parent_calc_shieldings[k]
            offsets.append(exp_val - parent_raw)

    if not offsets:
        return raw_shifts

    delta_template = float(np.mean(offsets))
    return raw_shifts + delta_template

class SHIFTBayesianFitter:
    def __init__(self, workspace_path: str) -> None:
        self.workspace = Path(workspace_path)
        self.registry_path = self.workspace / "cochem_shift_registry.json"
        self.registry = self._load_registry()
        self.engine: Optional[Any] = None

    def _load_registry(self) -> Dict[str, Any]:
        """Loads the registry to check tier and available files."""
        if not self.registry_path.exists():
            raise FileNotFoundError("SHIFT registry missing. Cannot initiate fitting.")
        with open(self.registry_path, "r") as f:
            return json.loads(f.read())

    def _load_theoretical_priors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the Boltzmann-averaged shifts to serve as the MCMC origin point."""
        npz_path = self.workspace / "shift_tensors.npz"
        if not npz_path.exists():
            raise FileNotFoundError("Theoretical tensors not found. Run Stage 3.0 first.")
        
        data = np.load(npz_path)
        avg_shifts = data['avg_shifts']
        
        product_class = self.registry.get("product_class", "PRODUCT_A")
        parent_exp = self.registry.get("parent_exp_shifts")
        parent_calc = self.registry.get("parent_calc_shieldings")
        ref_shielding = self.registry.get("ref_shielding", 184.0)

        if product_class in ["PRODUCT_B", "PRODUCT_C"] and parent_exp and parent_calc:
            calc_shieldings = ref_shielding - avg_shifts
            avg_shifts = apply_template_anchoring(calc_shieldings, ref_shielding, parent_exp, parent_calc)
        logger.info(f"⚓ Applied Product Class {product_class} Template Anchoring to chemical shifts.")
        
        # Flatten the shift array and initialize a blank J-matrix guess for the prior
        n_spins = len(avg_shifts)
        initial_j_matrix = np.zeros((n_spins, n_spins))
        
        return avg_shifts, initial_j_matrix

    def _log_prior(self, theta: np.ndarray, n_spins: int) -> float:
        """
        Applies rigid Bayesian boundaries to prevent non-physical wandering.
        theta layout: [shift_0, ..., shift_n, J_01, J_02, ..., J_nn]
        """
        shifts = theta[:n_spins]
        j_couplings = theta[n_spins:]
        
        if np.any(shifts < -10.0) or np.any(shifts > 250.0):
            return -np.inf
        if np.any(j_couplings < -100.0) or np.any(j_couplings > 500.0):
            return -np.inf
            
        return 0.0

    def _log_likelihood(self, theta: np.ndarray, n_spins: int, exp_spectrum: np.ndarray) -> float:
        """
        SHIFT-05: Validates J-flat dimension before reshaping into symmetric matrix.
        SHIFT-07: Cross-correlation spectral overlap loss function preserving multiplet structure.
        """
        shifts = jnp.array(theta[:n_spins])
        j_flat = theta[n_spins:]
        
        # SHIFT-05: Unchecked array dimension check
        expected_j_len = n_spins * (n_spins - 1) // 2
        if len(j_flat) != expected_j_len:
            raise ValueError(f"Invalid J-coupling vector length {len(j_flat)}. Expected {expected_j_len} for N={n_spins}.")
            
        j_matrix = np.zeros((n_spins, n_spins))
        idx = 0
        for i in range(n_spins):
            for j in range(i + 1, n_spins):
                j_matrix[i, j] = j_flat[idx]
                j_matrix[j, i] = j_flat[idx]
                idx += 1
                
        # 1. Hardware-Accelerated Hamiltonian Diagonalization
        eigenvalues, _ = self.engine.solve_hamiltonian(shifts, jnp.array(j_matrix))
        
        # 2. SHIFT-07: Cross-correlation Lorentzian overlap loss function
        synthetic_transitions = np.abs(np.diff(np.array(eigenvalues)))
        
        if len(exp_spectrum) == 0:
            return 0.0
            
        # Compute cross-correlation match score between synthetic transitions and experimental peaks
        grid = np.linspace(0.0, 200.0, min(1000, len(exp_spectrum)))
        synth_dens = np.zeros_like(grid)
        for tr in synthetic_transitions:
            synth_dens += 1.0 / (1.0 + ((grid - tr) / 0.5)**2)
            
        exp_dens = np.zeros_like(grid)
        for tr in exp_spectrum[:len(grid)]:
            exp_dens += 1.0 / (1.0 + ((grid - tr) / 0.5)**2)
            
        residual = float(np.sum((synth_dens - exp_dens)**2))
        return -0.5 * residual

    def _log_probability(self, theta: np.ndarray, n_spins: int, exp_spectrum: np.ndarray) -> float:
        """Combines the Prior and Likelihood for the MCMC sampler."""
        lp = self._log_prior(theta, n_spins)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta, n_spins, exp_spectrum)

    def execute_fitting(self, steps: int = 2000, n_walkers: int = 64) -> Path:
        """Initializes the MCMC ensemble, runs the burn-in, and samples the posterior."""
        logger.info("🧬 Initializing MCMC Bayesian Optimization...")
        
        tier = self.registry.get("tier", "T1-r2SCAN-3c")
        tier_upper = tier.upper()
        # Bypass MCMC completely when experimental spectrum is absent or tier is T1 / BRONZE
        if "T1" in tier_upper or "BRONZE" in tier_upper or "dx_hash" not in self.registry:
            raise RuntimeError("No experimental JCAMP-DX data found (Screening tier or missing data). Bayesian fitting requires experimental data. Use TheoreticalSpectrumEmitter instead.")

        # SHIFT-17: Raise explicit RuntimeError if JAX missing
        if not HAS_JAX:
            raise RuntimeError("JAX and JAXSpinHamiltonian are required for Bayesian fitting. Please install JAX.")


        avg_shifts, initial_j = self._load_theoretical_priors()
        n_spins = len(avg_shifts)
        
        # Initialize the JAX Engine
        self.engine = JAXSpinHamiltonian(str(self.workspace), n_spins=n_spins)
        
        # Prepare the parameter vector (Theta)
        j_flat = initial_j[np.triu_indices(n_spins, k=1)]
        theta_0 = np.concatenate([avg_shifts, j_flat])
        n_dim = len(theta_0)
        
        # Jiggle the starting positions of the walkers around the theoretical guess
        rng = np.random.default_rng(42)
        pos = theta_0 + 1e-4 * rng.standard_normal((n_walkers, n_dim))
        
        # Ingest real experimental target spectrum
        exp_spec_file = self.workspace / "exp_spectrum.npy"
        if not exp_spec_file.exists():
            raise FileNotFoundError(f"Experimental spectrum file {exp_spec_file} not found. Cannot proceed with Bayesian fitting.")
        exp_spectrum = np.load(exp_spec_file)
        
        import emcee
        backend_path = self.workspace / "mcmc_chains.h5"
        backend = emcee.backends.HDFBackend(str(backend_path))
        backend.reset(n_walkers, n_dim)
        
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, self._log_probability, 
            args=(n_spins, exp_spectrum), backend=backend
        )
        
        logger.info(f"🏃‍♂️ Running {n_walkers} walkers for {steps} steps...")
        sampler.run_mcmc(pos, steps, progress=True)
        
        flat_samples = sampler.get_chain(discard=int(steps*0.2), thin=15, flat=True)
        medians = np.median(flat_samples, axis=0)
        
        results = {
            "optimized_shifts_ppm": medians[:n_spins].tolist(),
            "optimized_j_couplings_hz": medians[n_spins:].tolist(),
            "mcmc_steps": steps,
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction))
        }
        
        output_file = self.workspace / "optimized_parameters.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"✅ MCMC Fitting Complete. Parameters saved to {output_file.name}")
        return output_file
def calculate_artifact_sha256(filepath: str | Path) -> str:
    """Calculates SHA-256 hash of a computational artifact."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Artifact file not found: {filepath}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()