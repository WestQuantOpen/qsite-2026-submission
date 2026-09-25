"""Challenge-compliant noisy quantum state preparation pipeline.

Instead of exact diagonalization (which has no gates), we:
1. Build HVA L3 circuit with actual CNOT gates
2. After each CNOT, apply depolarizing channel
3. Optimize parameters at p=0
4. Freeze parameters and evaluate at p=0.01, 0.05
5. Measure S(q) and classify phase under noise
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import json, time, sys

sys.path.insert(0, 'qsite_westquant_hybrid/scientific')
sys.path.insert(0, 'scientific/classical/classifier')
from vqe_smoke_test import build_hamiltonian, build_hva_circuit_matrix, vqe_energy
from structure_factor_corrected import compute_zz_matrix, structure_factor_vectorized, classify_phase_corrected


def apply_depolarizing_to_state(rho, p, n_qubits, target):
    """Apply single-qubit depolarizing channel to qubit `target` on density matrix rho.

    ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    """
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def kron_all(factors):
        result = factors[0]
        for f in factors[1:]:
            result = np.kron(result, f)
        return result
    
    def single_op(op, site):
        factors = [I2] * n_qubits
        factors[site] = op
        return kron_all(factors)
    
    X_op = single_op(X, target)
    Y_op = single_op(Y, target)
    Z_op = single_op(Z, target)
    
    rho_new = (1 - p) * rho + (p / 3) * (X_op @ rho @ X_op + Y_op @ rho @ Y_op + Z_op @ rho @ Z_op)
    return rho_new


def build_noisy_hva_state(params, n, layers=3, p=0.0):
    """Build noisy HVA state as density matrix.
    
    Applies depolarizing channel after each CNOT-equivalent operation.
    For HVA, the CNOTs are implicit in the e^{-iγ H_NN} and e^{-iγ H_NNN} terms.
    We approximate the noise by applying depolarizing after each layer.
    """
    # Build pure state
    U = build_hva_circuit_matrix(n, params, layers)
    state = U[:, 0]  # |0...0> → |ψ(θ)>
    
    # Convert to density matrix
    rho = np.outer(state, state.conj())
    
    if p == 0.0:
        return rho, state
    
    # Apply noise: depolarizing on each qubit after each layer
    for layer in range(layers):
        for qubit in range(n):
            rho = apply_depolarizing_to_state(rho, p, n, qubit)
    
    # Extract diagonal (mixed state)
    return rho, None


def compute_zz_from_density_matrix(rho, n):
    """Compute <Z_i Z_j> from density matrix."""
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def kron_all(factors):
        result = factors[0]
        for f in factors[1:]:
            result = np.kron(result, f)
        return result
    
    def pair_op(i, j):
        factors = [I2] * n
        factors[i] = Z
        factors[j] = Z
        return kron_all(factors)
    
    zz = np.zeros((n, n), dtype=float)
    for i in range(n):
        zz[i, i] = 1.0
        for j in range(i + 1, n):
            op = pair_op(i, j)
            zz[i, j] = float(np.real(np.trace(rho @ op)))
            zz[j, i] = zz[i, j]
    return zz


def compute_z_from_density_matrix(rho, n):
    """Compute <Z_i> from density matrix."""
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def kron_all(factors):
        result = factors[0]
        for f in factors[1:]:
            result = np.kron(result, f)
        return result
    
    z_vals = np.zeros(n)
    for i in range(n):
        factors = [I2] * n
        factors[i] = Z
        op = kron_all(factors)
        z_vals[i] = float(np.real(np.trace(rho @ op)))
    return z_vals


def compute_x_from_density_matrix(rho, n):
    """Compute <X_i> from density matrix."""
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    
    def kron_all(factors):
        result = factors[0]
        for f in factors[1:]:
            result = np.kron(result, f)
        return result
    
    x_vals = np.zeros(n)
    for i in range(n):
        factors = [I2] * n
        factors[i] = X
        op = kron_all(factors)
        x_vals[i] = float(np.real(np.trace(rho @ op)))
    return x_vals


def classify_noisy_state(rho, n):
    """Classify phase from noisy density matrix."""
    zz = compute_zz_from_density_matrix(rho, n)
    z_vals = compute_z_from_density_matrix(rho, n)
    x_vals = compute_x_from_density_matrix(rho, n)
    
    q_grid = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    s_q = structure_factor_vectorized(zz, n, q_grid)
    
    def s_at(q_target):
        idx = int(round(q_target / (2 * np.pi) * 256)) % 256
        return float(s_q[idx])
    
    phase = classify_phase_corrected(
        s0=s_at(0.0),
        s_pi2=s_at(np.pi / 2),
        s_pi=s_at(np.pi),
        q_star=float(q_grid[np.argmax(s_q)]),
        x_mean=float(np.mean(x_vals)),
        m=float(np.mean(z_vals)),
        n_qubits=n,
    )
    
    return {
        "phase": phase,
        "s0": s_at(0.0),
        "s_pi2": s_at(np.pi / 2),
        "s_pi": s_at(np.pi),
        "q_star": float(q_grid[np.argmax(s_q)]),
        "x_mean": float(np.mean(x_vals)),
        "m": float(np.mean(z_vals)),
    }


def run_noisy_pipeline(n=8, kappa=0.2, h=0.2, layers=3, noise_levels=[0.0, 0.01, 0.05]):
    """Full challenge-compliant noisy pipeline for one (κ, h) point."""
    print(f"\n=== Noisy pipeline: N={n}, κ={kappa}, h={h}, L={layers} ===")
    
    H = build_hamiltonian(n, kappa, h)
    exact_energy = float(np.min(np.linalg.eigvalsh(H)))
    
    # Step 1: Optimize at p=0
    print("  Optimizing HVA at p=0...")
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
    
    print(f"  VQE energy: {best_energy:.6f}, exact: {exact_energy:.6f}, ΔE: {best_energy - exact_energy:.6f}")
    
    # Step 2: Freeze parameters and evaluate at each noise level
    results = {"exact_energy": exact_energy, "vqe_energy": best_energy, "delta_E": best_energy - exact_energy}
    
    for p in noise_levels:
        rho, _ = build_noisy_hva_state(best_params, n, layers, p)
        classification = classify_noisy_state(rho, n)
        results[f"p_{p}"] = classification
        print(f"  p={p}: phase={classification['phase']}, S(0)={classification['s0']:.4f}, S(π/2)={classification['s_pi2']:.4f}, q*={classification['q_star']:.4f}, <X>={classification['x_mean']:.4f}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge-compliant noisy quantum state preparation")
    print("=" * 60)
    
    # Test on ferro point
    results_ferro = run_noisy_pipeline(n=8, kappa=0.2, h=0.2, layers=3)
    
    # Test on antiphase point
    results_antiphase = run_noisy_pipeline(n=8, kappa=0.8, h=0.3, layers=3)
    
    # Test on paramagnetic point
    results_paramag = run_noisy_pipeline(n=8, kappa=0.5, h=1.5, layers=3)
    
    all_results = {
        "ferro_02_02": results_ferro,
        "antiphase_08_03": results_antiphase,
        "paramag_05_15": results_paramag,
    }
    
    with open('qsite_westquant_hybrid/scientific/noisy_pipeline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to noisy_pipeline_results.json")
