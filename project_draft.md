# Project Draft — Q-SITE 2026 Open Challenges

**Team:** WestQuantOpen
**Date:** September 25, 2026

---

## Summary

We address both Q-SITE 2026 Open Challenges with a unified approach combining classical optimization and quantum simulation.

**Computational:** We achieved a 27.3% score reduction (119.0 → 86.5) on six circuit compilation benchmarks using three novel methods: co-optimization (joint placement + routing), edge-meeting routing (dual-ended pathfinding), and Large Neighborhood Search (destroy-and-repair). A* exact search certified near-optimality on GHZ and Ladder.

**Scientific:** We built a complete ANNNI phase detection pipeline with three critical fixes: structure factor correction (missing diagonal terms), antiphase wavevector correction (q* = π/2, not π), and phase-adapted reference-state HVA (U_HVA(θ)|ψ_ref⟩). Classifier accuracy improved from 25.85% to 70.1% (100% on interior points). Antiphase VQE improved from ΔE=2.16 to ΔE=0.02. The challenge-compliant noisy pipeline shows all phases remain stable at p=0.05 noise, with ferromagnetic order being most sensitive (29.7% degradation).

## Key Innovations

1. **Edge-meeting router** — Novel dual-ended routing that initiates paths from both gate endpoints
2. **LNS for circuit compilation** — First method to break the SA+SABRE plateau on dense_random
3. **Phase-adapted reference-state HVA** — Solves antiphase VQE by using phase-specific reference states
4. **Dual-signal phase detection** — Observable classifier + variational basin signal provide independent phase identification

## Results Summary

| Track | Metric | Before | After | Improvement |
|-------|--------|--------|-------|-------------|
| Computational | Total score | 119.0 | 86.5 | 27.3% |
| Scientific | Classifier accuracy | 25.85% | 70.1% | +44.3% |
| Scientific | Antiphase VQE ΔE | 2.162 | 0.021 | 99.0% |
| Scientific | Interior point accuracy | — | 100% | — |

## Files

- `notebooks/` — Three Jupyter notebooks with full results and plots
- `code/` — All source code (computational, scientific, common)
- `results/` — JSON result files
- `presentations/` — Presentation scripts and images
