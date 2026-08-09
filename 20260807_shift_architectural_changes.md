# CoChem-SHIFT: Architectural Changes (2026-08-07)

## 1. ZORA Relativistic GIAO Shielding
**Target File:** `shift_core/giao.py`
**Required Architectural Change:**
- SHIFT must automatically implement the Zero-Order Regular Approximation (ZORA) for relativistic scalar corrections when predicting NMR shieldings for heavy nuclei. The GIAO method is strictly enforced for origin invariance.

## 2. AIMD Thermal Averaging
**Target File:** `shift_core/averaging.py`
**Required Architectural Change:**
- A single static geometry is invalid for fluxional NMR. SHIFT must extract 100+ frames from KINETIC's AIMD trajectory, calculate the GIAO tensor for each, and thermally average the results to predict the true experimental line broadening and shift.

## 3. Basis Set Enforcement
**Target File:** `shift_core/basis_manager.py`
**Required Architectural Change:**
- Force the injection of specialized property-optimized basis sets (e.g., `pcSseg-2`) to ensure core-polarization functions adequately describe the magnetic field at the nucleus.
