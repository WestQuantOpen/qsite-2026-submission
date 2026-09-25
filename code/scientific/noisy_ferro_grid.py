"""Noisy pipeline on ferro region — multiple points with noise degradation analysis."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import json, time, sys

sys.path.insert(0, 'qsite_westquant_hybrid/scientific')
sys.path.insert(0, 'scientific/classical/classifier')
from vqe_smoke_test import build_hamiltonian, build_hva_circuit_matrix, vqe_energy
from noisy_pipeline import build_noisy_hva_state, classify_noisy_state


def run_point(n, kappa, h, layers=3, noise_levels=[0.0, 0.01, 0.05, 0.1]):
    H = build_hamiltonian(n, kappa, h)
    exact_energy = float(np.min(np.linalg.eigvalsh(H)))
    
    # Optimize at p=0
    n_params = 3 * layers
    best_energy = float('inf')
    best_params = None
    for trial in range(5):
        np.random.seed(42 + trial)
        x0 = np.random.uniform(0, 2 * np.pi, n_params)
        result = minimize(vqe_energy, x0, args=(n, H, layers), method='COBYLA', options={'maxiter': 500})
        if result.fun < best_energy:
            best_energy = float(result.fun)
            best_params = result.x
    
    # Evaluate at each noise level
    point_result = {
        "kappa": kappa, "h": h,
        "exact_energy": exact_energy,
        "vqe_energy": best_energy,
        "delta_E": best_energy - exact_energy,
        "noise": {}
    }
    
    for p in noise_levels:
        rho, _ = build_noisy_hva_state(best_params, n, layers, p)
        cls = classify_noisy_state(rho, n)
        point_result["noise"][str(p)] = cls
    
    return point_result


if __name__ == "__main__":
    print("=" * 60)
    print("Noisy pipeline — ferro region grid")
    print("=" * 60)
    
    # Ferro region: κ=0.0-0.4, h=0.0-1.0
    ferro_points = [
        (0.0, 0.0), (0.0, 0.3), (0.0, 0.6), (0.0, 1.0),
        (0.1, 0.0), (0.1, 0.3), (0.1, 0.6),
        (0.2, 0.0), (0.2, 0.2), (0.2, 0.4), (0.2, 0.6),
        (0.3, 0.0), (0.3, 0.3),
        (0.4, 0.0), (0.4, 0.3),
    ]
    
    results = []
    for kappa, h in ferro_points:
        t0 = time.time()
        r = run_point(n=8, kappa=kappa, h=h, layers=3)
        elapsed = time.time() - t0
        p0 = r["noise"]["0.0"]
        p5 = r["noise"]["0.05"]
        print(f"  κ={kappa:.1f} h={h:.1f}: ΔE={r['delta_E']:.4f} | p=0: {p0['phase']} S(0)={p0['s0']:.3f} | p=0.05: {p5['phase']} S(0)={p5['s0']:.3f} ({elapsed:.0f}s)")
        results.append(r)
    
    with open('qsite_westquant_hybrid/scientific/noisy_ferro_grid.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} points to noisy_ferro_grid.json")
    
    # Summary: compute ΔS(0) and ΔS(π/2) at p=0.05
    print("\n=== Noise degradation summary (p=0.05) ===")
    for r in results:
        p0 = r["noise"]["0.0"]
        p5 = r["noise"]["0.05"]
        delta_s0 = p0["s0"] - p5["s0"]
        delta_s_pi2 = p0["s_pi2"] - p5["s_pi2"]
        phase_stable = p0["phase"] == p5["phase"]
        print(f"  κ={r['kappa']:.1f} h={r['h']:.1f}: ΔS(0)={delta_s0:.4f} ΔS(π/2)={delta_s_pi2:.4f} phase_stable={phase_stable}")
