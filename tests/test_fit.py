"""
Unit tests for CoChem-SHIFT Stage 4.2 Bayesian Fitter.
"""

import os
import json
import tempfile
import pytest
import numpy as np

jax = pytest.importorskip("jax")
from cochem_shift_fit import SHIFTBayesianFitter

def test_fit_bronze_bypass():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "BRONZE"}, f)
            
        npz_path = os.path.join(tmpdir, "shift_tensors.npz")
        np.savez(npz_path, avg_shifts=np.array([1.5, 3.2, 7.1]))
        
        fitter = SHIFTBayesianFitter(tmpdir)
        out_json = fitter.execute_fitting()
        
        assert os.path.exists(out_json)
        with open(out_json, "r") as f:
            data = json.load(f)
            assert data["bypassed"] is True
            assert len(data["optimized_shifts_ppm"]) == 3
