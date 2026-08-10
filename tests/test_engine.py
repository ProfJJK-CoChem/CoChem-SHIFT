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

def test_v4_orca_command_builder():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "test_mol", "tier": "T2-PBE0-D4"}, f)
            
        dispatcher = SHIFTPhysicsDispatcher(tmpdir)
        
        # T1 Tier
        cmd_t1 = dispatcher._build_orca_command(tier="T1-r2SCAN-3c")
        assert "r2SCAN-3c" in cmd_t1
        assert "%mace" not in cmd_t1

        # T2 Tier
        cmd_t2 = dispatcher._build_orca_command(tier="T2-PBE0-D4")
        assert "PBE0-D4 pcSseg-2" in cmd_t2
        assert "%mace" not in cmd_t2

        # T3 Tier
        cmd_t3 = dispatcher._build_orca_command(tier="T3-pcSseg-3")
        assert "PBE0-D4 pcSseg-3" in cmd_t3

        # Deuterium Basis Upgrade
        cmd_d = dispatcher._build_orca_command(tier="T2-PBE0-D4", is_deuterium=True)
        assert "pcSseg-3" in cmd_d


def test_dispatcher_tms_reference_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "cochem_shift_registry.json")
        with open(reg_path, "w") as f:
            json.dump({"mol_name": "test_mol", "tier": "T2-PBE0-D4"}, f)
            
        xyz_path = os.path.join(tmpdir, "test_mol.xyz")
        with open(xyz_path, "w") as f:
            f.write("2\nTest\nC 0.0 0.0 0.0\nH 0.0 0.0 1.0\n")
            
        dispatcher = SHIFTPhysicsDispatcher(tmpdir)
        res = dispatcher.dispatch(xyz_path, solvent="Water")
        
        assert os.path.exists(os.path.join(tmpdir, "TMS_ideal.xyz"))
        assert os.path.exists(os.path.join(tmpdir, "TMS_reference_nmr.inp"))
        assert os.path.exists(os.path.join(tmpdir, "test_mol_nmr.inp"))
        
        with open(os.path.join(tmpdir, "test_mol_nmr.inp"), "r") as f:
            content = f.read()
            assert "PBE0-D4 pcSseg-2" in content
            assert "%mace" not in content

