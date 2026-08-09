"""
Unit tests for CoChem-SHIFT Stage 5.0 Publication Exporter.
"""

import os
import json
import tempfile
import pytest
import numpy as np
from cochem_shift_report import SHIFTPublicationExporter, sanitize_latex

def test_sanitize_latex():
    assert sanitize_latex("benzene_13C") == "benzene\\_13C"
    assert sanitize_latex("100% & $test") == "100\\% \\& \\$test"

def test_bipartite_matching_scale():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "benzene", "tier": "GOLD"}, f)
            
        params_path = os.path.join(tmpdir, "optimized_parameters.json")
        with open(params_path, "w") as f:
            json.dump({"optimized_shifts_ppm": [128.5, 130.1], "optimized_j_couplings_hz": [5.0]}, f)
            
        exporter = SHIFTPublicationExporter(tmpdir)
        exp_peaks = np.array([128.0, 131.0])
        calc_peaks = np.array([128.5, 130.1])
        
        # 13C shifts > 10 ppm should match within max_cutoff=30.0
        matches = exporter._bipartite_matching(exp_peaks, calc_peaks, max_cutoff=30.0)
        assert len(matches) == 2
        
        exporter.export_artifacts()
        assert os.path.exists(os.path.join(tmpdir, "SHIFT_Interactive.html"))
        assert os.path.exists(os.path.join(tmpdir, "SHIFT_Publication.tex"))
