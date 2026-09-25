"""VQE smoke test using NumPy HVA simulator with SciPy optimization.

HVA: U(θ) = ∏_ℓ e^{-iβ_ℓ H_X} e^{-iγ_{ℓ,1} H_NN} e^{-iγ_{ℓ,2} H_NNN}

Since terms within each block commute, we implement matrix exponentials
directly in NumPy for robustness. No PennyLane optimizer dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import json, time

I2 = np.eye(2, dtype=np.complex128)
XMAT = np.array([[0, 1], [1, 0]], dtype=np.complex128)
ZMAT = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kron_all(factors):
    result = factors[0]
    for f in factors[1:]:
        result = np.kron(result, f)
    return result


def build_hamiltonian(n, kappa, h, j1=1.0, periodic=True):
    """Build ANNNI Hamiltonian as numpy matrix. Z/Z/X convention."""
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    limit_nn = n if periodic else n - 1
    limit_nnn = n if periodic else n - 2

    # NN: -J1 * Z_i Z_{i+1}
    for i in range(limit_nn):
        j = (i + 1) % n
        factors = [I2] * n
        factors[i] = ZMAT
        factors[j] = ZMAT
        H += -j1 * kron_all(factors)

    # NNN: +J1*kappa * Z_i Z_{i+2}
    for i in range(limit_nnn):
        j = (i + 2) % n
        factors = [I2] * n
        factors[i] = ZMAT
        factors[j] = ZMAT
        H += j1 * kappa * kron_all(factors)

    # Transverse field: -h * sum X_i
    for i in range(n):
        factors = [I2] * n
        factors[i] = XMAT
        H += -h * kron_all(factors)

    return H


def build_hva_circuit_matrix(n, params, layers=3):
    """Build HVA circuit matrix U(θ) in NumPy.

    U(θ) = ∏_ℓ e^{-iβ_ℓ H_X} e^{-iγ_{ℓ,1} H_NN} e^{-iγ_{ℓ,2} H_NNN}
    """
    dim = 2**n
    U = np.eye(dim, dtype=np.complex128)

    # Build Hamiltonian blocks
    H_X = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        factors = [I2] * n
        factors[i] = XMAT
        H_X += kron_all(factors)

    H_NN = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        j = (i + 1) % n
        factors = [I2] * n
        factors[i] = ZMAT
        factors[j] = ZMAT
        H_NN += kron_all(factors)

    H_NNN = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        j = (i + 2) % n
        factors = [I2] * n
        factors[i] = ZMAT
        factors[j] = ZMAT
        H_NNN += kron_all(factors)

    for l in range(layers):
        beta = params[3 * l]
        gamma1 = params[3 * l + 1]
        gamma2 = params[3 * l + 2]

        # e^{-i*gamma2*H_NNN}
        U = np.linalg.matrix_multiply(U, None) if False else U  # placeholder
        from scipy.linalg import expm
        U = expm(-1j * gamma2 * H_NNN) @ U
        U = expm(-1j * gamma1 * H_NN) @ U
        U = expm(-1j * beta * H_X) @ U

    return U


def vqe_energy(params, n, H, layers=3):
    """Compute VQE energy E(θ) = <0|U† H U|0>."""
    U = build_hva_circuit_matrix(n, params, layers)
    state = U[:, 0]  # |0...0> is first column
    return float(np.real(np.vdot(state, H @ state)))


def run_vqe(n, kappa, h, layers=3, max_iter=500):
    """Run VQE for one (κ, h) point."""
    H = build_hamiltonian(n, kappa, h)

    # Exact ground state energy
    exact_energy = float(np.min(np.linalg.eigvalsh(H)))

    # VQE optimization
    n_params = 3 * layers
    best_energy = float('inf')
    best_params = None

    for trial in range(5):
        np.random.seed(42 + trial)
        x0 = np.random.uniform(0, 2 * np.pi, n_params)

        result = minimize(
            vqe_energy, x0, args=(n, H, layers),
            method='COBYLA',
            options={'maxiter': max_iter, 'rhobeg': 0.5}
        )

        if result.fun < best_energy:
            best_energy = float(result.fun)
            best_params = result.x

    delta_E = best_energy - exact_energy

    return {
        'n': n,
        'kappa': kappa,
        'h': h,
        'layers': layers,
        'exact_energy': exact_energy,
        'vqe_energy': best_energy,
        'delta_E': delta_E,
        'params': best_params.tolist() if best_params is not None else None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("VQE SMOKE TEST — NumPy HVA + SciPy COBYLA")
    print("=" * 60)

    # Smoke test: N=4, κ=0.2, h=0.2
    print("\n--- Smoke test: N=4, κ=0.2, h=0.2 ---")
    t0 = time.time()
    result = run_vqe(n=4, kappa=0.2, h=0.2, layers=3, max_iter=500)
    elapsed = time.time() - t0

    print(f"  Exact energy: {result['exact_energy']:.6f}")
    print(f"  VQE energy:   {result['vqe_energy']:.6f}")
    print(f"  ΔE:           {result['delta_E']:.6f}")
    print(f"  Time:         {elapsed:.1f}s")

    if abs(result['delta_E']) < 0.1:
        print("  STATUS: ✓ PASS (ΔE < 0.1)")
    else:
        print("  STATUS: ✗ FAIL (ΔE >= 0.1)")

    # Test N=8
    print("\n--- Test: N=8, κ=0.2, h=0.2 ---")
    t0 = time.time()
    result8 = run_vqe(n=8, kappa=0.2, h=0.2, layers=3, max_iter=500)
    elapsed8 = time.time() - t0

    print(f"  Exact energy: {result8['exact_energy']:.6f}")
    print(f"  VQE energy:   {result8['vqe_energy']:.6f}")
    print(f"  ΔE:           {result8['delta_E']:.6f}")
    print(f"  Time:         {elapsed8:.1f}s")

    # Test at antiphase point
    print("\n--- Test: N=8, κ=0.8, h=0.3 (antiphase) ---")
    t0 = time.time()
    result_ap = run_vqe(n=8, kappa=0.8, h=0.3, layers=3, max_iter=500)
    elapsed_ap = time.time() - t0

    print(f"  Exact energy: {result_ap['exact_energy']:.6f}")
    print(f"  VQE energy:   {result_ap['vqe_energy']:.6f}")
    print(f"  ΔE:           {result_ap['delta_E']:.6f}")
    print(f"  Time:         {elapsed_ap:.1f}s")

    # Save results
    all_results = {
        'smoke_test_N4': result,
        'test_N8_ferro': result8,
        'test_N8_antiphase': result_ap,
    }
    with open('qsite_westquant_hybrid/scientific/vqe_smoke_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to vqe_smoke_test_results.json")
