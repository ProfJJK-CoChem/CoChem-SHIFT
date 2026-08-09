"""
Unit tests for CoChem-SHIFT Stage 1.0 Ingestion.
"""

import os
import json
import tempfile
import pytest
from cochem_shift_ingest import SHIFTIngestor

def test_ingestor_xyz_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_path, "w") as f:
            f.write("1\nComment\nH 0.0 0.0 0.0\n")
            
        ingestor = SHIFTIngestor(tmpdir)
        reg = ingestor.ingest_data(xyz_path)
        
        assert reg["mol_name"] == "mol"
        assert reg["tier"] == "BRONZE"
        assert os.path.exists(os.path.join(tmpdir, "cochem_shift_registry.json"))
