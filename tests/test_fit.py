"""
Unit tests for CoChem-SHIFT Stage 4.2 Bayesian Fitter.
"""

import os
import json
import tempfile
import pytest
import numpy as np

jax = pytest.importorskip("jax")
from cochem_shift_fit import SHIFTBayesianFitter, apply_template_anchoring

def test_fit_screening_bypass():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "T1-r2SCAN-3c", "product_class": "PRODUCT_A"}, f)
            
        npz_path = os.path.join(tmpdir, "shift_tensors.npz")
        np.savez(npz_path, avg_shifts=np.array([1.5, 3.2, 7.1]))
        
        fitter = SHIFTBayesianFitter(tmpdir)
        out_json = fitter.execute_fitting()
        
        assert os.path.exists(out_json)
        with open(out_json, "r") as f:
            data = json.load(f)
            assert data["bypassed"] is True
            assert len(data["optimized_shifts_ppm"]) == 3
            assert data["provenance_tag"] == "[D]"

def test_template_anchoring():
    ref_shielding = 184.0
    calc_shieldings = np.array([182.5, 180.8, 176.9]) # raw shifts = [1.5, 3.2, 7.1]
    parent_exp_shifts = {"0": 1.6, "1": 3.3}
    parent_calc_shieldings = {"0": 182.5, "1": 180.8}
    
    # parent raw shifts = [1.5, 3.2], exp = [1.6, 3.3] -> offsets = [+0.1, +0.1] -> mean offset = +0.1
    anchored_shifts = apply_template_anchoring(calc_shieldings, ref_shielding, parent_exp_shifts, parent_calc_shieldings)
    np.testing.assert_allclose(anchored_shifts, np.array([1.6, 3.3, 7.2]), rtol=1e-5)

