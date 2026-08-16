import streamlit as st
import subprocess
import os
import sys
import psutil
import atexit
import hashlib
from pathlib import Path

st.set_page_config(page_title="CoChem-SHIFT - Native Pipeline UI", layout="wide")

def kill_zombie_processes() -> None:
    target_procs = ['orca', 'xtb', 'mpi', 'crest']
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if any(target in name for target in target_procs):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

atexit.register(kill_zombie_processes)

st.title("🔬 CoChem-SHIFT Control Panel")
st.markdown("This UI executes raw, heavy mathematical payloads natively.")

with st.sidebar:
    st.header("Pipeline Configuration")
    target_smiles = st.text_input("Target SMILES", "CCO")
    run_mode = st.selectbox("Execution Mode", ["Fast", "Accurate"])

if st.button("🚀 Execute Default Pipeline"):
    with st.spinner(f"Triggering quantum physics executor for {target_smiles}..."):
        st.info("Initiating Physical Math Execution Pipeline...")
        
        module_dir = Path(__file__).resolve().parent
        tests_dir = module_dir / "tests"
        
        env = os.environ.copy()
        env["COCHEM_TARGET_H5"] = os.path.join(os.getcwd(), "landscape.h5")
        
        try:
            st.error("Native execution requires an active task queue and worker nodes. The mock pipeline execution has been removed to eradicate spoofing.")
        except Exception as e:
            st.error(f"Pipeline crashed during physical execution: {str(e)}")
            kill_zombie_processes()
