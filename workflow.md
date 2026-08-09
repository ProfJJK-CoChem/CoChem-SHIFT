# CoChem-SHIFT: Software Engineering Specification
**Target Phase:** Python Implementation

This document serves as the exact coding blueprint for the next LLM agent to construct the `CoChem-SHIFT` repository.

## 1. Directory & File Architecture
```text
CoChem-SHIFT/
├── shift_core/
│   ├── __init__.py
│   ├── dispatcher.py       # Entry point for BASE payload ingestion
│   ├── frame_extractor.py  # Parses AIMD trajectory from KINETIC
│   ├── giao.py             # ZORA relativistic shielding setup
│   └── averaging.py        # Thermal statistics and J-coupling parsing
├── tests/
│   ├── test_giao.py
│   └── test_averaging.py
├── requirements.txt        # h5py, numpy, scipy
└── README.md
```

## 2. File-by-File Blueprint

### `shift_core/frame_extractor.py`
- **Purpose:** Isolates geometric snapshots from the HDF5 tensor.
- **Functions:**
  - `def subsample_trajectory(aimd_tensor: np.ndarray, n_frames: int = 100) -> np.ndarray:`
    - *Returns:* Equidistant geometric frames, actively discarding frames with unphysical bond distances.

### `shift_core/giao.py`
- **Purpose:** Constructs the ORCA properties block.
- **Functions:**
  - `def write_nmr_block(basis: str = "pcSseg-2", zora: bool = True) -> str:`
    - *Returns:* The ORCA string explicitly calling `%eprnmr` and `ZORA` if heavy atoms are present.

### `shift_core/averaging.py`
- **Purpose:** Parses 100 ORCA outputs and averages the tensors.
- **Functions:**
  - `def parse_shielding_tensor(orca_out: str) -> np.ndarray:`
    - *Returns:* The $3 \times 3$ isotropic magnetic shielding tensor.
  - `def calculate_shift(tensor_avg: float, tms_ref: float) -> float:`
    - *Returns:* The final relative $\delta$ ppm chemical shift.

## 3. Execution Data Flow (The Payload Trace)
1. **Payload Ingest & Prerequisites:** `dispatcher.py` receives an NMR prediction request. It immediately checks if an AIMD trajectory exists in `/kinetic/aimd_trajectory`. If not, it halts and pings `BASE` to trigger `KINETIC` first.
2. **Frame Extraction:** `frame_extractor.py` pulls 100 frames from the 298 K thermal trajectory.
3. **Job Generation:** `giao.py` generates 100 independent ORCA `.inp` files enforcing `pcSseg-2` and `ZORA`.
4. **HPC Fan-Out:** The 100 jobs are handed to `NODE` for simultaneous parallel execution.
5. **Tensor Averaging:** Once `NODE` completes, `averaging.py` parses all 100 properties files, extracting the GIAO tensors and Fermi-contact J-couplings. It averages them and calculates the variance ($\sigma$).
6. **Serialization:** The final $N \times N$ coupling matrices and averaged scalar ppm shifts are mapped into `/shift/nmr_tensors/`.

## 4. PyTest Roadmap
- **Test 1 (`test_giao.py`):** Assert that passing a coordinate file containing Platinum (Pt) forces `write_nmr_block` to return a string containing `ZORA`.
- **Test 2 (`test_averaging.py`):** Provide 100 mock tensor values with a known mean and standard deviation. Assert that the averaging algorithm reproduces the exact math and identifies highly fluxional protons (high $\sigma$).
