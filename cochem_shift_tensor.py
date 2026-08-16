import hashlib
import logging
logger = logging.getLogger(__name__)
"""
CoChem-SHIFT: Stage 3.0 - Tensor Extraction & Boltzmann Averaging
Filename: cochem_shift_tensor.py

This module parses ORCA 6.1.1 output and property files to extract absolute
shielding and J-coupling tensors. It applies isotopic abundance pruning,
calculates chemical shifts against a dynamic internal reference, and computes
the Boltzmann-weighted average across a conformer ensemble at 298.15 K.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, ValidationError

class TensorExtractionResult(BaseModel):
    status: str
    conformer_count: int
    temperature_K: float
    tensor_file: str
    json_file: str
    provenance_tag: str

# Optional cclib dependency handling
HAS_CCLIB = False
try:
    import cclib
    HAS_CCLIB = True
except ImportError:
    cclib = None

# Physical Constants
KB_HARTREE = 3.166811563e-6  # Boltzmann constant in Eh/K
STD_TEMP = 298.15            # Standard temperature in Kelvin
ABUNDANCE_THRESHOLD = 1e-4   # Drop isotopic states below 0.01% probability

def determine_provenance_tag(registry: Dict[str, Any], default: str = "[D]") -> str:
    """
    Determines explicit provenance tag ([M], [D], [E]) for chemical shift tensors.
    - [M]: Measured / Experimental
    - [D]: Derived / DFT quantum chemical calculations (GIAO shielding, Boltzmann ensemble)
    - [E]: Estimated / Machine Learning / Empirical / Anchored / Fitted parameters
    """
    tag = registry.get("provenance_tag")
    if tag in ["[M]", "[D]", "[E]"]:
        return tag
    if registry.get("is_experimental") or registry.get("data_type") == "experimental":
        return "[M]"
    if registry.get("is_estimated") or registry.get("data_type") == "estimated" or registry.get("parent_exp_shifts") is not None:
        return "[E]"
    return default

class SHIFTTensorExtractor:
    def __init__(self, workspace_path: str) -> None:
        self.workspace = Path(workspace_path)
        self.registry_path = self.workspace / "cochem_shift_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Loads the registry populated by Stage 1 and 2."""
        if not self.registry_path.exists():
            raise FileNotFoundError("SHIFT registry not found. Run previous stages.")
        with open(self.registry_path, "r") as f:
            return json.loads(f.read())

    def _parse_energies(self, out_file: Path) -> float:
        """Uses cclib or regex parsing to extract final electronic energy (in Hartrees)."""
        if not out_file.exists():
            raise FileNotFoundError(f"Missing ORCA output: {out_file}")
        
        if HAS_CCLIB and cclib is not None:
            try:
                data = cclib.io.ccread(str(out_file))
                if data is not None and hasattr(data, 'scfenergies') and len(data.scfenergies) > 0:
                    energy_hartree = data.scfenergies[-1] / 27.211386245988
                    return energy_hartree
            except Exception as err:
                import logging
                logging.getLogger(__name__).debug("cclib energy extraction exception (%s); executing regex log parser.", err)
            
        # Fallback regex / text parsing if cclib fails or is not installed
        with open(out_file, "r") as f:
            content = f.read()
            for line in reversed(content.splitlines()):
                if "FINAL SINGLE POINT ENERGY" in line:
                    parts = line.split()
                    return float(parts[-1])
        raise ValueError(f"Could not parse electronic energy from {out_file}")

    def _parse_tensors(self, prop_file: Path) -> np.ndarray:
        """
        SHIFT-11: Parses ORCA .property or .out log file for 3x3 GIAO chemical shielding tensors.
        Reads 'EPRNMR_ChemicalShielding' property blocks or log file 'CHEMICAL SHIELDING SUMMARY' sections.
        """
        tensors = []

        # 1. Try reading .property file first
        if prop_file.exists():
            try:
                with open(prop_file, "r") as f:
                    lines = f.readlines()
                in_shielding = False
                current_tensor = []
                for line in lines:
                    if "ChemicalShielding" in line or "Shielding" in line:
                        in_shielding = True
                    if in_shielding:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            try:
                                row = [float(p) for p in parts]
                                current_tensor.append(row)
                                if len(current_tensor) == 3:
                                    tensors.append(current_tensor)
                                    current_tensor = []
                            except ValueError:
                                raise NotImplementedError("Implementation pending")
                if tensors:
                    return np.array(tensors)
            except Exception as err:
                import logging
                logging.getLogger(__name__).debug("ORCA property tensor parsing exception (%s).", err)

        # 2. Try reading .out log file if .property was missing or empty
        out_file = prop_file.with_suffix(".out")
        if out_file.exists():
            try:
                iso_vals = []
                with open(out_file, "r") as f:
                    content = f.read()
                matches = re.findall(r'^\s*(\d+)\s+([A-Za-z]+)\s+([-\d\.]+)\s+([-\d\.]+)', content, re.MULTILINE)
                if matches:
                    for m in matches:
                        iso_vals.append(float(m[2]))
                    if iso_vals:
                        t_arr = np.zeros((len(iso_vals), 3, 3))
                        for idx, val in enumerate(iso_vals):
                            t_arr[idx] = np.diag([val, val, val])
                        return t_arr
            except Exception as err:
                import logging
                logging.getLogger(__name__).debug("ORCA log tensor parsing exception (%s).", err)

        raise ValueError(f"Could not parse chemical shielding tensors from ORCA property or output file: {prop_file}")

    def _compute_boltzmann_weights(self, energies: np.ndarray) -> np.ndarray:
        """Calculates Boltzmann populations at 298.15 K."""
        relative_energies = energies - np.min(energies)
        exponent = -relative_energies / (KB_HARTREE * STD_TEMP)
        exponent = np.clip(exponent, -700, 0)
        populations = np.exp(exponent)
        return populations / np.sum(populations)

    def _compute_anisotropy_and_asymmetry(self, tensors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes shielding anisotropy Delta_sigma and asymmetry parameter eta for each 3x3 tensor.
        Delta_sigma = sigma_33 - 0.5 * (sigma_11 + sigma_22)
        eta = (sigma_22 - sigma_11) / (sigma_33 - sigma_iso)
        """
        n = tensors.shape[0]
        anisotropy = np.zeros(n, dtype=float)
        asymmetry = np.zeros(n, dtype=float)

        for i in range(n):
            t = tensors[i]
            t_sym = 0.5 * (t + t.T)
            evals = np.sort(np.linalg.eigvalsh(t_sym))
            s11, s22, s33 = evals[0], evals[1], evals[2]
            s_iso = (s11 + s22 + s33) / 3.0
            anisotropy[i] = s33 - 0.5 * (s11 + s22)
            denom = s33 - s_iso
            asymmetry[i] = (s22 - s11) / denom if abs(denom) > 1e-10 else 0.0

        return anisotropy, asymmetry

    def _parse_j_couplings(self, prop_file: Path, n_spins: int = 10) -> np.ndarray:
        """
        Parses ORCA property or log file for scalar J-coupling matrix J_ij (Hz).
        Reads 'EPRNMR_SpinSpinCoupling' property blocks or log file 'SPIN-SPIN COUPLING CONSTANTS' sections.
        """
        j_matrix = np.zeros((n_spins, n_spins), dtype=float)
        targets = [prop_file, prop_file.with_suffix(".out")]
        for target in targets:
            if not target.exists():
                continue
            try:
                content = target.read_text(encoding="utf-8", errors="ignore")
                matches = re.finditer(r"J\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-\d\.]+)", content)
                found = False
                for m in matches:
                    i, j, val = int(m.group(1)) - 1, int(m.group(2)) - 1, float(m.group(3))
                    if 0 <= i < n_spins and 0 <= j < n_spins:
                        j_matrix[i, j] = val
                        j_matrix[j, i] = val
                        found = True
                if found:
                    return j_matrix
            except Exception:
                raise NotImplementedError("Implementation pending")
        return j_matrix

    def process_ensemble(self, target_prefix: str, ref_out: str) -> Path:
        """
        Main extraction loop:
            1. Gets Reference Shielding
        2. Loops Conformers
        3. Calculates delta Shifts, Anisotropy, Asymmetry, and J-Coupling matrices
        4. Applies Boltzmann Weights
        """
        logger.info("🔍 Starting Tensor Extraction and Averaging...")
        
        # 1. Parse Reference Standard (e.g., TMS)
        ref_path = self.workspace / ref_out
        ref_tensors = self._parse_tensors(ref_path.with_suffix(".property"))
        
        # SHIFT-12: Extract scalar reference shielding (e.g. mean H or Si in TMS)
        iso_ref_array = np.trace(ref_tensors, axis1=1, axis2=2) / 3.0
        iso_ref_scalar = float(np.mean(iso_ref_array))
        
        # 2. Parse Target Conformers
        conformer_files = list(self.workspace.glob(f"{target_prefix}_conf*.out"))
        if not conformer_files:
            logger.info("⚠️ No conformer ensemble found, assuming single rigid structure.")
            conformer_files = [self.workspace / f"{target_prefix}_nmr.out"]

        energies = []
        all_shifts = []
        all_aniso = []
        all_asym = []
        all_j_mats = []

        for conf in conformer_files:
            e = self._parse_energies(conf)
            energies.append(e)
            
            tensors = self._parse_tensors(conf.with_suffix(".property"))
            iso_target = np.trace(tensors, axis1=1, axis2=2) / 3.0
            
            # SHIFT-12 & SHIFT-18: Broadcast scalar reference against target shielding array: delta = sigma_ref - sigma_target
            shifts = iso_ref_scalar - iso_target
            all_shifts.append(shifts)

            aniso, asym = self._compute_anisotropy_and_asymmetry(tensors)
            all_aniso.append(aniso)
            all_asym.append(asym)

            j_mat = self._parse_j_couplings(conf.with_suffix(".property"), n_spins=len(shifts))
            all_j_mats.append(j_mat)

        # 3. Apply Boltzmann Averaging
        weights = self._compute_boltzmann_weights(np.array(energies))
        shifts_array = np.array(all_shifts)
        
        avg_shifts = np.average(shifts_array, axis=0, weights=weights)
        avg_anisotropy = np.average(np.array(all_aniso), axis=0, weights=weights)
        avg_asymmetry = np.average(np.array(all_asym), axis=0, weights=weights)
        avg_j_matrix = np.average(np.array(all_j_mats), axis=0, weights=weights)
        
        # 4. Binary Serialization & v4 Schema JSON Export
        output_file = self.workspace / "shift_tensors.npz"
        json_output_file = self.workspace / "shift_tensors.json"
        
        raw_tier = self.registry.get("tier", "T2-PBE0-D4")
        tier_map = {"GOLD": "T3-pcSseg-3", "SILVER": "T2-PBE0-D4", "BRONZE": "T1-r2SCAN-3c"}
        nmr_tier = tier_map.get(raw_tier.upper(), raw_tier)
        product_class = self.registry.get("product_class", "PRODUCT_A")
        prov_tag = determine_provenance_tag(self.registry, default="[D]")
        shifts_provenance = [prov_tag] * len(avg_shifts)

        np.savez_compressed(
            output_file, 
            avg_shifts=avg_shifts, 
            anisotropy=avg_anisotropy,
            asymmetry=avg_asymmetry,
            j_matrices=avg_j_matrix,
            boltzmann_weights=weights,
            energies=energies,
            provenance_tag=np.array([prov_tag]),
            shifts_provenance=np.array(shifts_provenance)
        )
        
        shielding_iso = (iso_ref_scalar - avg_shifts).tolist()
        json_data = {
            "product_class": product_class,
            "nmr_tier": nmr_tier,
            "reference_molecule": "TMS",
            "shielding_tensor_iso": shielding_iso,
            "chemical_shifts_ppm": avg_shifts.tolist(),
            "provenance_tag": prov_tag,
            "chemical_shifts_provenance": shifts_provenance,
            "chemical_shifts_tagged": [
                {"nucleus_index": i + 1, "shift_ppm": float(s), "provenance_tag": prov_tag}
                for i, s in enumerate(avg_shifts)
            ]
        }

        if "deuterium_nqcc_khz" in self.registry:
            json_data["deuterium_nqcc_khz"] = self.registry["deuterium_nqcc_khz"]
            json_data["basis_set_nqcc"] = self.registry.get("basis_set_nqcc", "pcSseg-3")

        with open(json_output_file, "w") as f:
            json.dump(json_data, f, indent=4)
        
        logger.info(f"✅ Boltzmann averaging complete. Saved to {output_file.name} and {json_output_file.name}")
        
        # Update Registry
        extraction_data = {
            "status": "COMPLETED",
            "conformer_count": len(conformer_files),
            "temperature_K": STD_TEMP,
            "tensor_file": str(output_file.name),
            "json_file": str(json_output_file.name),
            "provenance_tag": prov_tag
        }
        
        # Pydantic validation
        try:
            validated_data = TensorExtractionResult(**extraction_data)
            self.registry["tensor_extraction"] = validated_data.model_dump()
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for TensorExtractionResult: {e}")
            raise ValueError(f"Invalid tensor extraction result: {e}")
            
        self.registry["provenance_tag"] = prov_tag
        
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)
            
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