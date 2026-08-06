"""
CoChem-SHIFT: Stage 1.0 - Ingestion Validator & Provenance Lock
Filename: cochem_shift_ingest.py
"""

import json
import hashlib
import os
import nmrglue as ng
from typing import Dict, Any, Optional
from pathlib import Path

class SHIFTIngestor:
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.registry_path = self.workspace / "cochem_shift_registry.json"
        self.config = self._load_system_config()

    def _load_system_config(self) -> Dict[str, Any]:
        """Loads the authoritative environment registry."""
        config_file = Path("cochem_system_config.json")
        if not config_file.exists():
            raise FileNotFoundError("System registry missing! Run Stage 0.0.")
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
            return "GOLD"
        elif has_experimental:
            return "SILVER"
        return "BRONZE"

    def ingest_data(self, xyz_file: str, dx_file: Optional[str] = None) -> Dict[str, Any]:
        """Validates inputs and locks them into the SHIFT registry."""
        xyz_path = Path(xyz_file)
        
        # 1. Validation
        if not xyz_path.exists():
            raise FileNotFoundError(f"Input file {xyz_file} not found.")

        # 2. Provenance Generation
        registry_entry = {
            "mol_name": xyz_path.stem,
            "xyz_hash": self._generate_hash(xyz_path),
            "tier": "BRONZE",  # Default
            "status": "VALIDATED"
        }

        # 3. Handle Experimental Data
        if dx_file and Path(dx_file).exists():
            try:
                dic, data = ng.jcampdx.read(dx_file)
                registry_entry["dx_hash"] = self._generate_hash(Path(dx_file))
                registry_entry["tier"] = self.determine_tier(True, False)
            except Exception as e:
                print(f"⚠️ Warning: Failed to parse JCAMP-DX: {e}")

        # 4. Save to Registry
        self.workspace.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(registry_entry, f, indent=4)
        
        print(f"✅ Ingestion successful. Tier assigned: {registry_entry['tier']}")
        return registry_entry

# Example usage for integration:
# ingestor = SHIFTIngestor("./SHIFT_Workspace")
# ingestor.ingest_data("mol_conformer.xyz", "experiment.dx")