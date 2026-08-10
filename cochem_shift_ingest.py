"""
CoChem-SHIFT: Stage 1.0 - Ingestion Validator & Provenance Lock
Filename: cochem_shift_ingest.py
"""

import json
import hashlib
import os
from typing import Dict, Any, Optional
from pathlib import Path

# SHIFT-10: Handled optional nmrglue import gracefully
HAS_NMRGLUE = False
try:
    import nmrglue as ng
    HAS_NMRGLUE = True
except ImportError:
    ng = None

class SHIFTIngestor:
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.registry_path = self.workspace / "cochem_shift_registry.json"
        self.config = self._load_system_config()

    def _load_system_config(self) -> Dict[str, Any]:
        """Loads the authoritative environment registry."""
        config_file = Path("cochem_system_config.json")
        if not config_file.exists():
            # Return basic configuration if not present
            return {"shift_engine": {"workspace_path": str(self.workspace)}}
        with open(config_file, "r") as f:
            return json.load(f)

    def _generate_hash(self, file_path: Path) -> str:
        """Generates SHA-256 hash for provenance tracking."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def determine_tier(self, has_experimental: bool, has_high_tier_tensors: bool) -> str:
        """Determines the calculation tier based on available inputs."""
        if has_high_tier_tensors:
            return "T3-pcSseg-3"
        elif has_experimental:
            return "T2-PBE0-D4"
        return "T1-r2SCAN-3c"

    def ingest_data(
        self,
        xyz_file: str,
        dx_file: Optional[str] = None,
        product_class: str = "PRODUCT_A",
        parent_exp_shifts: Optional[Dict[str, float]] = None,
        parent_calc_shieldings: Optional[Dict[str, float]] = None,
        ref_shielding: Optional[float] = None,
        tier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validates inputs and locks them into the SHIFT registry."""
        xyz_path = Path(xyz_file)
        
        # 1. Validation
        if not xyz_path.exists():
            raise FileNotFoundError(f"Input file {xyz_file} not found.")

        # Validate Product Class
        valid_classes = ["PRODUCT_A", "PRODUCT_B", "PRODUCT_C"]
        if product_class not in valid_classes:
            raise ValueError(f"Invalid product_class '{product_class}'. Must be one of {valid_classes}.")

        # 2. Provenance Generation
        assigned_tier = tier if tier is not None else "T1-r2SCAN-3c"
        registry_entry = {
            "mol_name": xyz_path.stem,
            "xyz_hash": self._generate_hash(xyz_path),
            "tier": assigned_tier,
            "product_class": product_class,
            "status": "VALIDATED"
        }

        if parent_exp_shifts is not None:
            registry_entry["parent_exp_shifts"] = parent_exp_shifts
            if product_class == "PRODUCT_A":
                registry_entry["product_class"] = "PRODUCT_B"
        if parent_calc_shieldings is not None:
            registry_entry["parent_calc_shieldings"] = parent_calc_shieldings
        if ref_shielding is not None:
            registry_entry["ref_shielding"] = ref_shielding

        # 3. Handle Experimental Data (SHIFT-10: Safe nmrglue check)
        if dx_file and Path(dx_file).exists():
            dx_p = Path(dx_file)
            if HAS_NMRGLUE:
                try:
                    dic, data = ng.jcampdx.read(str(dx_p))
                    registry_entry["dx_hash"] = self._generate_hash(dx_p)
                    if tier is None:
                        registry_entry["tier"] = self.determine_tier(True, False)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to parse JCAMP-DX: {e}")
            else:
                print("⚠️ nmrglue package is missing. Storing file hash without metadata parsing.")
                registry_entry["dx_hash"] = self._generate_hash(dx_p)
                if tier is None:
                    registry_entry["tier"] = self.determine_tier(True, False)

        # 4. Save to Registry
        self.workspace.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(registry_entry, f, indent=4)
        
        print(f"✅ Ingestion successful. Tier assigned: {registry_entry['tier']}, Product Class: {registry_entry['product_class']}")
        return registry_entry