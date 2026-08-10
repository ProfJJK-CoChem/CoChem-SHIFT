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
        reg = ingestor.ingest_data(xyz_path, product_class="PRODUCT_A")
        
        assert reg["mol_name"] == "mol"
        assert reg["tier"] == "T1-r2SCAN-3c"
        assert reg["product_class"] == "PRODUCT_A"
        assert os.path.exists(os.path.join(tmpdir, "cochem_shift_registry.json"))

def test_determine_tier():
    ingestor = SHIFTIngestor(".")
    assert ingestor.determine_tier(has_experimental=False, has_high_tier_tensors=False) == "T1-r2SCAN-3c"
    assert ingestor.determine_tier(has_experimental=True, has_high_tier_tensors=False) == "T2-PBE0-D4"
    assert ingestor.determine_tier(has_experimental=True, has_high_tier_tensors=True) == "T3-pcSseg-3"

