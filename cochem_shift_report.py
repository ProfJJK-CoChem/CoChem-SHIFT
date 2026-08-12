import logging
logger = logging.getLogger(__name__)
"""
CoChem-SHIFT: Stage 5.0 - Bipartite Assignment & FAIR Reporting
Filename: cochem_shift_report.py

This module finalizes the CoChem-SHIFT pipeline by analyzing the Bayesian
posterior outputs. It utilizes SciPy's linear sum assignment to create a 
bipartite graph mapping between theoretical and experimental spectral peaks, 
generates an interactive offline Plotly dashboard, and writes a publication-
grade LaTeX table.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple  # SHIFT-20: Added Tuple import
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

_LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_LATEX_ESCAPE_REGEX = re.compile("|".join(re.escape(k) for k in _LATEX_REPLACEMENTS.keys()))

def sanitize_latex(text: str) -> str:
    """
    SHIFT-15: Comprehensive LaTeX string sanitizer using single-pass regex replacement.
    """
    if not text:
        return ""
    return _LATEX_ESCAPE_REGEX.sub(lambda match: _LATEX_REPLACEMENTS[match.group(0)], text)

def determine_provenance_tag(params: Dict[str, Any], registry: Dict[str, Any], default: str = "[D]") -> str:
    """
    Determines explicit provenance tag ([M], [D], [E]) for reporting artifacts.
    - [M]: Measured / Experimental
    - [D]: Derived / Quantum Chemical DFT calculations
    - [E]: Estimated / Machine Learning / Empirical / Anchored / Fitted parameters
    """
    tag = params.get("provenance_tag") or registry.get("provenance_tag")
    if tag in ["[M]", "[D]", "[E]"]:
        return tag
    if params.get("bypassed") is False or registry.get("is_estimated"):
        return "[E]"
    if registry.get("is_experimental") or registry.get("data_type") == "experimental":
        return "[M]"
    return default

class SHIFTPublicationExporter:
    def __init__(self, workspace_path: str) -> None:
        self.workspace = Path(workspace_path)
        self.registry_path = self.workspace / "cochem_shift_registry.json"
        self.params_path = self.workspace / "optimized_parameters.json"
        
        self.registry = self._load_json(self.registry_path)
        self.params = self._load_json(self.params_path)

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Safely loads required JSON configuration and result files."""
        if not file_path.exists():
            raise FileNotFoundError(f"Required file missing: {file_path.name}. Run previous stages.")
        with open(file_path, "r") as f:
            return json.loads(f.read())

    def _bipartite_matching(self, exp_peaks: np.ndarray, calc_peaks: np.ndarray, max_cutoff: float = 30.0) -> List[Tuple[int, int]]:
        """
        SHIFT-13: Uses Hungarian Algorithm to assign peaks with nucleus-aware threshold (default 30.0 ppm).
        Supports 13C chemical shift ranges (0-220 ppm) without prematurely rejecting valid matches.
        """
        exp_peaks = np.asarray(exp_peaks)
        calc_peaks = np.asarray(calc_peaks)
        
        if len(exp_peaks) == 0 or len(calc_peaks) == 0:
            return []
            
        cost_matrix = np.abs(exp_peaks[:, np.newaxis] - calc_peaks)
        
        # Prevent unphysical assignments beyond max_cutoff threshold
        cost_matrix[cost_matrix > max_cutoff] = 1e6 
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = [(r, c) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 1e6]
        return assignments

    def _generate_plotly_dashboard(self, shifts: List[float], j_couplings: List[float], line_width: float = 0.05) -> Path:
        """
        SHIFT-14: Configurable line width parameter for Lorentzian broadening in Plotly dashboard.
        """
        fig = go.Figure()
        
        raw_tier = self.params.get("nmr_tier") or self.registry.get("tier", "T1-r2SCAN-3c")
        tier_map = {"GOLD": "T3-pcSseg-3", "SILVER": "T2-PBE0-D4", "BRONZE": "T1-r2SCAN-3c"}
        tier = tier_map.get(str(raw_tier).upper(), str(raw_tier))
        product_class = self.params.get("product_class") or self.registry.get("product_class", "PRODUCT_A")
        prov_tag = determine_provenance_tag(self.params, self.registry, default="[D]")

        if len(shifts) > 0:
            x_axis = np.linspace(min(shifts) - 2, max(shifts) + 2, 1000)
            y_axis = np.zeros_like(x_axis)
            
            for shift in shifts:
                y_axis += 1.0 / (1.0 + ((x_axis - shift) / line_width)**2)
                
            fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name=f'Theoretical {prov_tag}', line=dict(color='blue')))
            fig.add_trace(go.Bar(x=shifts, y=[1]*len(shifts), name=f'Assigned Shifts {prov_tag}', marker_color='red', width=0.01))
        
        fig.update_layout(
            title=f"CoChem-SHIFT: Bayesian Optimized NMR Spectrum ({tier}, {product_class}, {prov_tag})",
            xaxis_title="Chemical Shift (ppm)",
            yaxis_title="Normalized Intensity",
            template="plotly_dark",
            xaxis=dict(autorange="reversed")
        )
        
        output_html = self.workspace / "SHIFT_Interactive.html"
        fig.write_html(str(output_html))
        return output_html

    def _generate_latex_report(self, shifts: List[float], j_couplings: List[float]) -> Path:
        """
        SHIFT-15: Compiles siunitx LaTeX table with sanitized special characters and provenance tags.
        """
        raw_name = self.registry.get("mol_name", "Unknown_Molecule")
        mol_name = sanitize_latex(raw_name)
        
        raw_tier = self.params.get("nmr_tier") or self.registry.get("tier", "T1-r2SCAN-3c")
        tier_map = {"GOLD": "T3-pcSseg-3", "SILVER": "T2-PBE0-D4", "BRONZE": "T1-r2SCAN-3c"}
        tier = sanitize_latex(tier_map.get(str(raw_tier).upper(), str(raw_tier)))
        
        product_class = sanitize_latex(self.params.get("product_class") or self.registry.get("product_class", "PRODUCT_A"))
        raw_prov_tag = determine_provenance_tag(self.params, self.registry, default="[D]")
        prov_tag = sanitize_latex(raw_prov_tag)
        shifts_prov = self.params.get("chemical_shifts_provenance") or self.params.get("shifts_provenance") or [raw_prov_tag] * len(shifts)
        
        latex_str = f"""\\documentclass[11pt, a4paper]{{article}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}

\\begin{{document}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Bayesian Optimized Chemical Shifts and $J$-couplings for {mol_name} at the {tier} Tier ({product_class}, Provenance: {prov_tag}).}}
\\begin{{tabular}}{{l S[table-format=2.4] l}}
\\toprule
{{Nucleus Index}} & {{Chemical Shift (ppm)}} & {{Provenance}} \\\\
\\midrule
"""
        for i, shift in enumerate(shifts):
            s_prov = sanitize_latex(shifts_prov[i]) if i < len(shifts_prov) else prov_tag
            latex_str += f"{i+1} & {shift:.4f} & {s_prov} \\\\\n"
            
        latex_str += """\\bottomrule
\\end{tabular}
\\end{table}

\\end{document}
"""
        output_tex = self.workspace / "SHIFT_Publication.tex"
        with open(output_tex, "w") as f:
            f.write(latex_str)
            
        return output_tex

    def export_artifacts(self) -> None:
        """Main orchestration method to generate all FAIR deliverables."""
        logger.info("📊 Generating FAIR Publication Artifacts...")
        
        shifts = self.params.get("optimized_shifts_ppm", [])
        j_couplings = self.params.get("optimized_j_couplings_hz", [])
        
        # 1. Interactive Dashboard
        html_path = self._generate_plotly_dashboard(shifts, j_couplings)
        logger.info(f"✅ Interactive Dashboard generated: {html_path.name}")
        
        # 2. LaTeX Publication Table
        tex_path = self._generate_latex_report(shifts, j_couplings)
        logger.info(f"✅ LaTeX mechanism table generated: {tex_path.name}")
        
        # 3. Update Registry
        prov_tag = determine_provenance_tag(self.params, self.registry, default="[D]")
        self.registry["provenance_tag"] = prov_tag
        self.registry["status"] = "PIPELINE_COMPLETE"
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)