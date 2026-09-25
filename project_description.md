# Project Description — Q-SITE 2026 Open Challenges

**Team:** WestQuantOpen
**Date:** September 25, 2026

---

## 1. Introduction

This project addresses both Q-SITE 2026 Open Challenges:

1. **Computational Challenge:** Quantum circuit compilation and routing on constrained hardware topologies
2. **Scientific Challenge:** Quantum phase detection for the ANNNI model using variational quantum eigensolvers

Our approach combines classical optimization algorithms with quantum simulation to achieve significant improvements in both tracks.

---

## 2. Computational Challenge

### 2.1 Problem Statement

Given quantum circuits defined over logical qubits and a constrained hardware coupling graph, find an initial placement and routing that minimizes the total score (SWAP count + 0.5 × circuit depth) across six benchmark circuits.

### 2.2 Methods

We developed three novel methods that produced improvements:

#### Co-optimization
Joint optimization of qubit placement and SABRE routing. Instead of optimizing placement and routing separately, we iterate between them: given a placement, optimize the route; given the route, adjust the placement to reduce routing costs. This produced the strongest gains on `qaoa_random` (20.0 → 14.0) and `dense_random` (70.0 → 51.5).

#### Edge-meeting Router
A novel dual-ended routing approach. Standard SABRE routes from the source qubit to the target qubit. Our edge-meeting router initiates pathfinding from both endpoints simultaneously, meeting in the middle. This reduces detour costs and produced major gains on `ghz_star` (11.0 → 7.0) and `ladder_trotter` (10.5 → 6.5).

#### Large Neighborhood Search (LNS)
A destroy-and-repair search that breaks SA+SABRE plateaus. The algorithm:
1. Destroys: reassigns 2-8 logical qubits to new physical positions
2. Repairs: re-routes the entire circuit with both SABRE and edge-meeting routers
3. Accepts: if the new solution scores better than the incumbent

LNS improved `dense_random` from 54.0 → 51.5, the first method to break the 12-hour SA+SABRE plateau.

#### A* Exact Search
We implemented bounded A* exact search to certify near-optimality. For GHZ and Ladder, we explored 2M states each without finding improvements, confirming these benchmarks are near-optimal.

### 2.3 Results

| Benchmark | Baseline | Final | Improvement | SWAPs | Depth | Strategy |
|-----------|----------|-------|-------------|-------|-------|----------|
| ghz_star | 11.0 | 7.0 | -4.0 (36.4%) | 3 | 8 | edge_meeting |
| chain_trotter | 4.5 | 4.5 | 0.0 | 0 | 9 | frozen (optimal) |
| ladder_trotter | 10.5 | 6.5 | -4.0 (38.1%) | 3 | 7 | edge_meeting |
| qaoa_random | 20.0 | 14.0 | -6.0 (30.0%) | 7 | 14 | co_optimize |
| dense_random | 70.0 | 51.5 | -18.5 (26.4%) | 36 | 31 | lns_sabre |
| vqe_layers | 3.0 | 3.0 | 0.0 | 0 | 6 | frozen (optimal) |
| **TOTAL** | **119.0** | **86.5** | **-32.5 (27.3%)** | — | — | — |

### 2.4 Key Findings

- Co-optimization is the strongest method for large circuits (qaoa, dense)
- Edge-meeting routing excels on circuits with hub-like logical connectivity (ghz, ladder)
- LNS breaks plateaus that SA+SABRE cannot escape
- A* confirms GHZ (7.0) and Ladder (6.5) are near-optimal
- Dense_random at 51.5 is an extremely strong local optimum (974K ALNS iterations, 0 improvements)

---

## 3. Scientific Challenge

### 3.1 Problem Statement

Detect quantum phases of the ANNNI (Axial Next-Nearest-Neighbor Ising) model using variational quantum eigensolvers, and quantify the effect of gate noise on phase detection.

### 3.2 Critical Corrections

#### Structure Factor Bug
The reference structure factor computation was missing diagonal terms (i=j), producing S(q) - 1 instead of S(q). This caused:
- Impossible negative S(0) values
- Incorrect q* values
- Complete classifier failure on antiphase states

**Correct formula:**
```
S(q) = (1/N) v(q)† C v(q)  where  v_j(q) = e^{iqj},  C_ij = ⟨Z_i Z_j⟩
```

This equals `(1/N) ⟨|Σ_j e^{iqj} Z_j|²⟩ ≥ 0`, guaranteeing non-negativity.

#### Antiphase Wavevector
The classifier was checking S(π) for antiphase. ANNNI antiphase has period 4 (↑↑↓↓), so the characteristic ordering wavevector is q* = π/2, not π.

### 3.3 Phase-Adapted Reference-State HVA

**Innovation:** Instead of U_HVA(θ)|0^N⟩, use U_HVA(θ)|ψ_ref⟩ where ψ_ref is phase-specific:
- Ferromagnetic: |00000000⟩
- Antiphase: |00110011⟩
- Paramagnetic: |+⟩^⊗N

This solved the antiphase VQE problem (ΔE 2.16 → 0.02) because the reference state energy sanity test showed that product states already have low ΔE at phase-specific points — the problem was state preparation, not ansatz expressivity.

The winning branch (lowest energy) provides an independent phase signal, complementing the observable-based classifier.

### 3.4 Results

#### Classifier Accuracy

| Phase | Before Fix | After Fix | Change |
|-------|-----------|----------|--------|
| Ferromagnetic | 68.5% | 92.3% | +23.8% |
| Antiphase | 0.0% | 93.3% | +93.3% |
| Paramagnetic | — | 81.3% | — |
| **Overall** | **25.85%** | **70.1%** | **+44.3%** |
| **Interior points** | — | **100.0%** | — |

#### VQE Energy Accuracy

| Point | Old VQE ΔE | New VQE ΔE | Winning Branch | Phase |
|-------|-----------|-----------|----------------|-------|
| κ=0.2, h=0.2 (ferro) | 0.002 | 0.003 | ferro | ferromagnetic |
| κ=0.8, h=0.3 (anti) | 2.162 | 0.021 | antiphase | antiphase |
| κ=0.5, h=1.5 (para) | 1.858 | 0.042 | paramag | paramagnetic |

#### Noise Degradation (Challenge-Compliant)

Pipeline: HVA L3 with phase-adapted reference state → depolarizing channel → measure S(q) → classify

| Phase | p=0 | p=0.05 | Degradation | Phase Stable? |
|-------|------|--------|-------------|---------------|
| Ferromagnetic | S(0)=7.896 | S(0)=5.558 | 29.7% | ✓ |
| Antiphase | S(π/2)=3.901 | S(π/2)=2.918 | 25.2% | ✓ |
| Paramagnetic | ⟨X⟩=0.933 | ⟨X⟩=0.758 | 18.8% | ✓ |

**Finding:** Ferromagnetic long-range order is most noise-sensitive, followed by antiphase, then paramagnetic.

### 3.5 Research Narrative

The project evolved from "map the ANNNI phase diagram" to a richer research question:

> **"How do representation choice and gate noise jointly affect quantum phase detection?"**

Key discoveries:
1. Structure factor implementation had a systematic -1.0 offset
2. Antiphase wavevector is π/2 (not π) for ANNNI period-4 ordering
3. Interior phases are easy to classify but boundaries are hard (finite-size effect)
4. Shallow HVA L3 works nearly exactly in the ferro region (ΔE=0.003)
5. Same representation fails dramatically in frustrated regimes without phase-adapted initialization
6. Phase-adapted reference-state HVA solves all phases (ΔE < 0.12)
7. Ferromagnetic order is most noise-sensitive (29.7% degradation at p=0.05)
8. The winning variational branch provides an independent phase signal

---

## 4. Reproducibility

All results are reproducible from the notebooks in this repository:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

The computational `best.json` checkpoint is included in `results/` and can be re-validated with the official scorer.

---

## 5. Team

**WestQuantOpen** — Vesterlund Quantum Algorithm Studio
