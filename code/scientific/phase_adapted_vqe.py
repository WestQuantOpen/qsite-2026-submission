"""Phase-adapted reference-state HVA.

Instead of U_HVA(θ)|0^N⟩, use U_HVA(θ)|ψ_ref⟩ where ψ_ref is phase-specific:
  - Ferro: |00000000⟩
  - Antiphase: |00110011⟩
  - Paramagnetic: |+⟩^8

The winning branch (lowest energy) provides additional phase information.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import json, time, sys

sys.path.insert(0, 'qsite_westquant_hybrid/scientific')
from vqe_smoke_test import build_hamiltonian, build_hva_circuit_matrix, vqe_energy


def build_reference_states(n):
    """Build phase-specific reference states."""
    ferro = np.zeros(2**n, dtype=np.complex128)
    ferro[0] = 1.0  # |00000000>
    
    antiphase = np.zeros(2**n, dtype=np.complex128)
    antiphase[0b00110011] = 1.0  # |00110011>
    
    paramag = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)  # |+>^8
    
    return {
        'ferro': ferro,
        'antiphase': antiphase,
        'paramag': paramag,
    }


def hva_energy_with_ref(params, n, H, ref_state, layers=3):
    """Compute E(θ) = <ψ_ref|U† H U|ψ_ref>."""
    U = build_hva_circuit_matrix(n, params, layers)
    state = U @ ref_state
    return float(np.real(np.vdot(state, H @ state)))


def run_phase_adapted_vqe(n, kappa, h, layers=3, max_iter=1000, n_trials=3):
    """Run phase-adapted HVA with reference-state initialization."""
    H = build_hamiltonian(n, kappa, h)
    exact_energy = float(np.min(np.linalg.eigvalsh(H)))
    ref_states = build_reference_states(n)
    
    n_params = 3 * layers
    best_energy = float('inf')
    best_params = None
    best_branch = None
    best_ref = None
    branch_results = {}
    
    for ref_name, ref_state in ref_states.items():
        branch_best = float('inf')
        branch_best_params = None
        
        for trial in range(n_trials):
            np.random.seed(42 + trial)
            x0 = np.random.uniform(0, 2 * np.pi, n_params)
            
            # COBYLA
            result = minimize(
                hva_energy_with_ref, x0, args=(n, H, ref_state, layers),
                method='COBYLA',
                options={'maxiter': max_iter, 'rhobeg': 0.5}
            )
            if result.fun < branch_best:
                branch_best = float(result.fun)
                branch_best_params = result.x
            
            # Powell fallback
            result2 = minimize(
                hva_energy_with_ref, x0, args=(n, H, ref_state, layers),
                method='Powell',
                options={'maxiter': max_iter // 2}
            )
            if result2.fun < branch_best:
                branch_best = float(result2.fun)
                branch_best_params = result2.x
        
        branch_results[ref_name] = {
            'energy': branch_best,
            'delta_E': branch_best - exact_energy,
        }
        
        if branch_best < best_energy:
            best_energy = branch_best
            best_params = branch_best_params
            best_branch = ref_name
            best_ref = ref_state
    
    # Compute observables for winning branch
    U = build_hva_circuit_matrix(n, best_params, layers)
    winning_state = U @ best_ref
    
    # Classify the winning state
    sys.path.insert(0, 'scientific/classical/classifier')
    from structure_factor_corrected import classify_phase_from_state_corrected
    classification = classify_phase_from_state_corrected(winning_state, n)
    
    return {
        'n': n, 'kappa': kappa, 'h': h, 'layers': layers,
        'exact_energy': exact_energy,
        'vqe_energy': best_energy,
        'delta_E': best_energy - exact_energy,
        'best_branch': best_branch,
        'predicted_phase': classification['phase'],
        's0': classification['s0'],
        's_pi2': classification['s_pi2'],
        'q_star': classification['q_star'],
        'x_mean': classification['x_mean'],
        'branch_results': branch_results,
        'params': best_params.tolist() if best_params is not None else None,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Phase-adapted reference-state HVA")
    print("U_HVA(θ)|ψ_ref⟩  instead of  U_HVA(θ)|0^N⟩")
    print("=" * 70)
    
    test_points = [
        (0.2, 0.2, 'ferro'),
        (0.5, 0.5, 'floating'),
        (0.8, 0.3, 'antiphase'),
        (1.0, 0.5, 'antiphase'),
        (0.2, 1.5, 'paramag'),
        (0.5, 1.5, 'paramag'),
        (0.8, 1.5, 'paramag'),
    ]
    
    results = []
    for kappa, h, label in test_points:
        t0 = time.time()
        r = run_phase_adapted_vqe(n=8, kappa=kappa, h=h, layers=3, max_iter=1000, n_trials=3)
        elapsed = time.time() - t0
        
        status = '✓' if abs(r['delta_E']) < 0.1 else ('~' if abs(r['delta_E']) < 0.5 else '✗')
        print(f"\n  κ={kappa} h={h} ({label}):")
        print(f"    ΔE={r['delta_E']:.4f} branch={r['best_branch']} phase={r['predicted_phase']} {status} ({elapsed:.0f}s)")
        print(f"    S(0)={r['s0']:.3f} S(π/2)={r['s_pi2']:.3f} q*={r['q_star']:.3f} <X>={r['x_mean']:.3f}")
        for bn, br in r['branch_results'].items():
            print(f"    {bn}: ΔE={br['delta_E']:.4f}")
        
        results.append(r)
    
    with open('qsite_westquant_hybrid/scientific/phase_adapted_vqe_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to phase_adapted_vqe_results.json")
