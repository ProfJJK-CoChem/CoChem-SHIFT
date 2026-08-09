"""
Unit tests for CoChem-SHIFT Stage 3.0 Tensor Extraction and Boltzmann Averaging.
"""

import os
import json
import tempfile
import pytest
import numpy as np
from cochem_shift_tensor import SHIFTTensorExtractor

def test_tensor_extraction_and_averaging():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "BRONZE"}, f)
            
        ref_out = os.path.join(tmpdir, "TMS_reference_nmr.out")
        with open(ref_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -500.0\n")
            
        target_out = os.path.join(tmpdir, "target_nmr.out")
        with open(target_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -300.0\n")
            
        extractor = SHIFTTensorExtractor(tmpdir)
        out_npz = extractor.process_ensemble("target", "TMS_reference_nmr.out")
        
        assert os.path.exists(out_npz)
        with np.load(out_npz) as data:
            assert "avg_shifts" in data
            assert len(data["avg_shifts"]) == 10
