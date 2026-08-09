# CoChem-SHIFT

**CoChem-SHIFT** is the Dynamic NMR Spectroscopy engine of the extended CoChem suite.

It is responsible for:
- Implementing Gauge-Independent Atomic Orbitals (GIAO) alongside Zero-Order Regular Approximation (ZORA) relativistic corrections to predict exact magnetic shielding tensors.
- Mandating property-optimized core-polarization basis sets (e.g., `pcSseg-2`).
- Extracting 100+ frames from `CoChem-KINETIC`'s AIMD trajectory to perform dynamic thermal averaging, moving past the failure of rigid-rotor approximations for fluxional molecules.
- Delivering highly interactive 1D Plotly UI elements with Peak-to-Atom dynamic mapping and JCAMP-DX (`.jdx`) experimental overlays.

## Usage
Please refer to the authoritative `CoChem_Master_User_Manual.md` located in the `CoChem-BASE` repository for full execution instructions across the entire pipeline.