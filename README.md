# Q-SITE 2026 Challenge Submission — WestQuantOpen

## Overview

This repository contains our solutions to the Q-SITE 2026 Open Challenges, addressing both the **Computational** and **Scientific** tracks.

### Computational Challenge: Quantum Circuit Compilation & Routing

We achieved a **27.3% score reduction** (119.0 → 86.5) on six benchmark circuits using three novel methods:

1. **Co-optimization** — Joint placement + SABRE routing optimization
2. **Edge-meeting router** — Dual-ended routing that initiates paths from both gate endpoints
3. **Large Neighborhood Search (LNS)** — Destroy-and-repair search that breaks SA+SABRE plateaus

A* exact search confirmed GHZ and Ladder benchmarks are near-optimal (2M states explored each, no improvement found).

### Scientific Challenge: ANNNI Phase Detection with VQE

We built a complete quantum phase detection pipeline for the ANNNI model with three critical fixes:

1. **Structure factor correction** — Fixed a systematic -1.0 offset from missing diagonal terms
2. **Antiphase wavevector correction** — q* = π/2 (period 4: ↑↑↓↓), not π
3. **Phase-adapted reference-state HVA** — U_HVA(θ)|ψ_ref⟩ instead of U_HVA(θ)|0^N⟩

Results:
- Classifier accuracy: 25.85% → 70.1% (100% on interior points)
- Antiphase VQE: ΔE 2.16 → 0.02
- Challenge-compliant noisy pipeline with quantitative degradation data

## Repository Structure

```
qsite-2026-submission/
├── README.md                          # This file
├── project_description.md             # Detailed project description
├── project_draft.md                   # Project draft summary
├── requirements.txt                    # Python dependencies
├── notebooks/
│   ├── 1_computational_challenge.ipynb # Computational track notebook
│   ├── 2_scientific_challenge.ipynb    # Scientific track notebook
│   └── 3_results_summary.ipynb         # Results summary with plots
├── code/
│   ├── computational/                  # Circuit compilation algorithms
│   ├── scientific/                     # Phase detection & VQE code
│   └── common/                         # Shared utilities
├── results/                            # JSON results files
└── presentations/
    ├── computational_script.md         # Presentation script (Computational)
    ├── scientific_script.md            # Presentation script (Scientific)
    └── images/                         # Generated plots
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/

# Reproduce computational results
python code/computational/alns_dense.py

# Reproduce scientific results
python code/scientific/phase_adapted_vqe.py
```

## Team

**WestQuantOpen** — Vesterlund Quantum Algorithm Studio

## License

MIT
