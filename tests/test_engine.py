"""
Unit tests for CoChem-SHIFT Stage 2.0 Physics Dispatcher.
"""

import os
import json
import tempfile
import pytest
from cochem_shift_engine import SHIFTPhysicsDispatcher, ATOMIC_NUMBERS

def test_atomic_numbers_completeness():
    assert ATOMIC_NUMBERS["H"] == 1
    assert ATOMIC_NUMBERS["C"] == 6
    assert ATOMIC_NUMBERS["Se"] == 34
    assert ATOMIC_NUMBERS["I"] == 53
    assert len(ATOMIC_NUMBERS) >= 100

def test_dispatcher_tms_reference_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "test_mol", "tier": "BRONZE"}, f)
            
        xyz_path = os.path.join(tmpdir, "test_mol.xyz")
        with open(xyz_path, "w") as f:
            f.write("2\nTest\nC 0.0 0.0 0.0\nH 0.0 0.0 1.0\n")
            
        dispatcher = SHIFTPhysicsDispatcher(tmpdir)
        res = dispatcher.dispatch(xyz_path, solvent="Water")
        
        assert os.path.exists(os.path.join(tmpdir, "TMS_ideal.xyz"))
        assert os.path.exists(os.path.join(tmpdir, "TMS_reference_nmr.inp"))
        assert os.path.exists(os.path.join(tmpdir, "test_mol_nmr.inp"))
        
        # Verify model name spelling correction (Thesseus_NMR)
        with open(os.path.join(tmpdir, "test_mol_nmr.inp"), "r") as f:
            content = f.read()
            assert "Thesseus_NMR" in content
