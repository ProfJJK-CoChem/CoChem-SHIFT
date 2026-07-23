# **CoChem-SHIFT: NMR Bayesian Fitting & Synthesis**

## **Overview**

**CoChem-SHIFT** is an advanced Nuclear Magnetic Resonance (NMR) emulator. Raw chemical shift tensors from DFT are notoriously difficult to map directly to experimental parts-per-million (ppm) without scaling.

SHIFT solves this via a **Tiered Bayesian MCMC Engine**. It ingests theoretical shielding tensors (calculated via GIAO) and scales them against empirical reference sets using Markov Chain Monte Carlo. For heavy atoms where scalar relativistic effects dominate, SHIFT automatically escalates to X2C (Exact Two-Component) Hamiltonians.

## **Scientific & Technical Trade-offs**

* **Out-of-Core MCMC Execution:** Bayesian fitting using hundreds of walkers over 10,000 steps generates massive coordinate arrays. SHIFT routes the emcee backend directly to an HDF5 disk cache (mcmc\_chains.h5). You trade SSD write cycles for a completely flat RAM profile, ensuring the Python kernel never crashes during an overnight fit.  
* **Tiered Ingestion Strategy:** Running full relativistic CCSD(T) NMR on a 50-atom molecule is practically impossible. SHIFT allows a "Gold/Silver/Bronze/Raw" ingestion logic. If you only need a quick structural validation (Bronze), it routes to rapid DFT (e.g., r2scan-3c). It sacrifices chemical precision for rapid triage, only escalating to the heavy computational tiers when user-defined variance thresholds are violated.

## **Installation & Setup**

git clone \[https://github.com/CoChem/CoChem-SHIFT.git\](https://github.com/CoChem/CoChem-SHIFT.git)  
cd CoChem-SHIFT

## **How to Run**

1. **Intake and Tier Routing:**  
   python cochem\_shift\_registry.py \--input target.xyz \--tier Silver  
2. **Execute Relativistic Dispatch:**  
   python cochem\_shift\_dispatcher.py  
   *(Automatically handles X2C or DKH2 routing if elements heavier than Krypton are detected).*  
3. **Run Bayesian Synthesizer:**  
   python cochem\_shift\_mcmc.py  
   *(Generates the final probability-weighted NMR spectrum).*