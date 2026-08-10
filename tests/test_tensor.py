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
            json.dump({"mol_name": "target", "tier": "T2-PBE0-D4", "product_class": "PRODUCT_A"}, f)
            
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
            assert "anisotropy" in data
            assert "asymmetry" in data
            assert "j_matrices" in data
            assert "provenance_tag" in data
            assert str(data["provenance_tag"][0]) in ["[M]", "[D]", "[E]"]
            assert len(data["avg_shifts"]) == 10

        json_path = os.path.join(tmpdir, "shift_tensors.json")
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            json_data = json.load(f)
            assert json_data["product_class"] == "PRODUCT_A"
            assert json_data["nmr_tier"] == "T2-PBE0-D4"
            assert json_data["provenance_tag"] == "[D]"
            assert "chemical_shifts_ppm" in json_data
            assert "shielding_tensor_iso" in json_data

def test_shift_ir_generator():
    from cochem_shift_ir import SHIFTIRSpectrumGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SHIFTIRSpectrumGenerator(tmpdir)
        html_p, tex_p = gen.generate_ir_artifacts()
        assert html_p.exists()
        assert tex_p.exists()
        assert "FORWARD IR" in html_p.read_text(encoding="utf-8").upper()

