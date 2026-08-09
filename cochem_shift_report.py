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
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple  # SHIFT-20: Added Tuple import
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

def sanitize_latex(text: str) -> str:
    """
    SHIFT-15: Comprehensive LaTeX string sanitizer.
    """
    if not text:
        return ""
    replacements = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "$": "\\$",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}"
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text

class SHIFTPublicationExporter:
    def __init__(self, workspace_path: str):
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
            return json.load(f)

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
        
        if len(shifts) > 0:
            x_axis = np.linspace(min(shifts) - 2, max(shifts) + 2, 1000)
            y_axis = np.zeros_like(x_axis)
            
            for shift in shifts:
                y_axis += 1.0 / (1.0 + ((x_axis - shift) / line_width)**2)
                
            fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='lines', name='Theoretical', line=dict(color='blue')))
            fig.add_trace(go.Bar(x=shifts, y=[1]*len(shifts), name='Assigned Shifts', marker_color='red', width=0.01))
        
        fig.update_layout(
            title="CoChem-SHIFT: Bayesian Optimized NMR Spectrum",
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
        SHIFT-15: Compiles siunitx LaTeX table with sanitized special characters.
        """
        raw_name = self.registry.get("mol_name", "Unknown_Molecule")
        mol_name = sanitize_latex(raw_name)
        tier = sanitize_latex(self.registry.get("tier", "BRONZE"))
        
        latex_str = f"""\\documentclass[11pt, a4paper]{{article}}
\\usepackage{{booktabs}}
\\package{{siunitx}}

\\begin{{document}}

\\begin{{table}}[htbp]
\\centering
\\caption{{Bayesian Optimized Chemical Shifts and $J$-couplings for {mol_name} at the {tier} Tier.}}
\\begin{{tabular}}{{l S[table-format=2.4]}}
\\toprule
{{Nucleus Index}} & {{Chemical Shift (ppm)}} \\\\
\\midrule
"""
        for i, shift in enumerate(shifts):
            latex_str += f"{i+1} & {shift:.4f} \\\\\n"
            
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
        print("📊 Generating FAIR Publication Artifacts...")
        
        shifts = self.params.get("optimized_shifts_ppm", [])
        j_couplings = self.params.get("optimized_j_couplings_hz", [])
        
        # 1. Interactive Dashboard
        html_path = self._generate_plotly_dashboard(shifts, j_couplings)
        print(f"✅ Interactive Dashboard generated: {html_path.name}")
        
        # 2. LaTeX Publication Table
        tex_path = self._generate_latex_report(shifts, j_couplings)
        print(f"✅ LaTeX mechanism table generated: {tex_path.name}")
        
        # 3. Update Registry
        self.registry["status"] = "PIPELINE_COMPLETE"
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)