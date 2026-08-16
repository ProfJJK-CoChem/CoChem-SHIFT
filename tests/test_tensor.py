import hashlib
from typing import Any, Dict, List, Optional
"""
Unit tests for CoChem-SHIFT Stage 3.0 Tensor Extraction and Boltzmann Averaging.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
import numpy as np
from cochem_shift_tensor import SHIFTTensorExtractor

def test_tensor_extraction_and_averaging() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "T2-PBE0-D4", "product_class": "PRODUCT_A"}, f)
            
        ref_out = os.path.join(tmpdir, "TMS_reference_nmr.out")
        with open(ref_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -500.0\n")
            f.write("CHEMICAL SHIELDING SUMMARY\n")
            for i in range(1, 11):
                f.write(f"{i} H 184.0 184.0\n")
            
        target_out = os.path.join(tmpdir, "target_nmr.out")
        with open(target_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -300.0\n")
            f.write("CHEMICAL SHIELDING SUMMARY\n")
            for i in range(1, 11):
                f.write(f"{i} H 31.5 31.5\n")
            
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
            json_data = json.loads(f.read())
            assert json_data["product_class"] == "PRODUCT_A"
            assert json_data["nmr_tier"] == "T2-PBE0-D4"
            assert json_data["provenance_tag"] == "[D]"
            assert "chemical_shifts_ppm" in json_data
            assert "shielding_tensor_iso" in json_data

def test_shift_ir_generator() -> None:
    from cochem_shift_ir import SHIFTIRSpectrumGenerator
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = SHIFTIRSpectrumGenerator(tmpdir)
        html_p, tex_p = gen.generate_ir_artifacts()
        assert html_p.exists()
        assert tex_p.exists()
        assert "FORWARD IR" in html_p.read_text(encoding="utf-8").upper()

def test_tensor_provenance_tags_m_d_e() -> None:
    for expected_tag in ["[M]", "[D]", "[E]"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
            with open(reg_path, "w") as f:
                json.dump({"mol_name": "target", "tier": "T2-PBE0-D4", "product_class": "PRODUCT_A", "provenance_tag": expected_tag}, f)
                
            ref_out = os.path.join(tmpdir, "TMS_reference_nmr.out")
            with open(ref_out, "w") as f:
                f.write("FINAL SINGLE POINT ENERGY -500.0\n")
                f.write("CHEMICAL SHIELDING SUMMARY\n")
                for i in range(1, 11):
                    f.write(f"{i} H 184.0 184.0\n")
                
            target_out = os.path.join(tmpdir, "target_nmr.out")
            with open(target_out, "w") as f:
                f.write("FINAL SINGLE POINT ENERGY -300.0\n")
                f.write("CHEMICAL SHIELDING SUMMARY\n")
                for i in range(1, 11):
                    f.write(f"{i} H 31.5 31.5\n")
                
            extractor = SHIFTTensorExtractor(tmpdir)
            out_npz = extractor.process_ensemble("target", "TMS_reference_nmr.out")
            
            with np.load(out_npz) as data:
                assert str(data["provenance_tag"][0]) == expected_tag
                assert "shifts_provenance" in data
                assert str(data["shifts_provenance"][0]) == expected_tag

            json_path = os.path.join(tmpdir, "shift_tensors.json")
            with open(json_path, "r") as f:
                json_data = json.loads(f.read())
                assert json_data["provenance_tag"] == expected_tag
                assert json_data["chemical_shifts_provenance"][0] == expected_tag
                assert json_data["chemical_shifts_tagged"][0]["provenance_tag"] == expected_tag

def test_tensor_spoofing_fallback_removed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "T2-PBE0-D4", "product_class": "PRODUCT_A"}, f)
            
        ref_out = os.path.join(tmpdir, "TMS_reference_nmr.out")
        with open(ref_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -500.0\n")
            # No tensors written, so it will fail parsing
        
        extractor = SHIFTTensorExtractor(tmpdir)
        with pytest.raises(ValueError, match="Could not parse chemical shielding tensors"):
            extractor.process_ensemble("target", "TMS_reference_nmr.out")

def test_energy_spoofing_fallback_removed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "target", "tier": "T2-PBE0-D4", "product_class": "PRODUCT_A"}, f)
            
        ref_out = os.path.join(tmpdir, "TMS_reference_nmr.out")
        with open(ref_out, "w") as f:
            f.write("FINAL SINGLE POINT ENERGY -500.0\n")
            f.write("CHEMICAL SHIELDING SUMMARY\n")
            for i in range(1, 11):
                f.write(f"{i} H 184.0 184.0\n")

        target_out = os.path.join(tmpdir, "target_nmr.out")
        with open(target_out, "w") as f:
            # Missing energy log line
            f.write("CHEMICAL SHIELDING SUMMARY\n")
            for i in range(1, 11):
                f.write(f"{i} H 31.5 31.5\n")
        
        extractor = SHIFTTensorExtractor(tmpdir)
        with pytest.raises(ValueError, match="Could not parse electronic energy"):
            extractor.process_ensemble("target", "TMS_reference_nmr.out")

def test_pydantic_rejections_tensor() -> None:
    from cochem_shift_tensor import TensorExtractionResult
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        # Missing required status
        TensorExtractionResult(conformer_count=1, temperature_K=298.15, tensor_file="file", json_file="file.json", provenance_tag="[D]")
        
    with pytest.raises(ValidationError):
        # Invalid type for temperature_K
        TensorExtractionResult(status="COMPLETED", conformer_count=1, temperature_K="invalid", tensor_file="file", json_file="file.json", provenance_tag="[D]")


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