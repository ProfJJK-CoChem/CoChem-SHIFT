import os
import json
import tempfile
import pytest
import numpy as np

from cochem_shift_engine import SHIFTPhysicsDispatcher
from cochem_shift_fit import SHIFTBayesianFitter
from cochem_shift_ingest import SHIFTIngestor, ShiftRegistry
from pydantic import ValidationError
from cochem_shift_tensor import SHIFTTensorExtractor

def test_engine_generates_tightscf_and_defgrid3() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "test_mol", "tier": "T3-pcSseg-3"}, f)
        dispatcher = SHIFTPhysicsDispatcher(tmpdir)
        cmd_t3 = dispatcher._build_orca_command(tier="T3-pcSseg-3", is_opt=True)
        assert "TightSCF DEFGRID3" in cmd_t3
        assert "%geom" in cmd_t3
        assert "InHess XTB2" in cmd_t3
        assert "TolMaxG 1e-5" in cmd_t3

def test_fit_screening_bypass() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "T1-r2SCAN-3c", "product_class": "PRODUCT_A"}, f)
        npz_path = os.path.join(tmpdir, "shift_tensors.npz")
        np.savez(npz_path, avg_shifts=np.array([1.5, 3.2, 7.1]))
        fitter = SHIFTBayesianFitter(tmpdir)
        with pytest.raises(RuntimeError, match="No experimental JCAMP-DX data found"):
            fitter.execute_fitting()

def test_ingest_pydantic_rejections() -> None:
    with pytest.raises(ValidationError):
        ShiftRegistry(xyz_hash="hash", tier="T1-r2SCAN-3c", product_class="PRODUCT_A", status="VALIDATED")
    with pytest.raises(ValidationError):
        ShiftRegistry(mol_name="test", xyz_hash="hash", tier="T1-r2SCAN-3c", product_class="PRODUCT_A", status="VALIDATED", ref_shielding="invalid_string")

def test_tensor_spoofing_fallback_removed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "T2-PBE0-D4", "product_class": "PRODUCT_A"}, f)
        ref_out = os.path.join(tmpdir, "TMS_reference_nmr.out")
        with open(ref_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -500.0\n")
        extractor = SHIFTTensorExtractor(tmpdir)
        with pytest.raises(ValueError, match="Could not parse chemical shielding tensors"):
            extractor.process_ensemble("target", "TMS_reference_nmr.out")
