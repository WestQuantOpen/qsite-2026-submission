"""Corrected structure factor and phase classifier for the ANNNI model.

CRITICAL FIXES:
1. S(q) = (1/N) v(q)^† C v(q) where C_ij = <Z_i Z_j> (INCLUDING diagonal)
   - Previous implementation excluded diagonal, giving S(q) - 1 instead of S(q)
   - S(q) >= 0 always (it's < |sum_j e^{iqj} Z_j|^2 > / N)

2. ANNNI antiphase has period 4 (↑↑↓↓), so q* = π/2, NOT π
   - Previous classifier checked S(π) for antiphase — WRONG
   - Correct: check S(π/2) for antiphase

3. Classifier uses physics-based features:
   - Ferromagnetic: q* ≈ 0, S(0) large
   - Antiphase: q* ≈ π/2, S(π/2) large
   - Paramagnetic: <X> high, S(q) ~ 1 (flat)
   - Floating: q* at incommensurate position
"""
from __future__ import annotations

import numpy as np


def compute_zz_matrix(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute the full <Z_i Z_j> correlation matrix from a state vector."""
    from observables_compat import _pair_operator

    zz = np.zeros((n_qubits, n_qubits), dtype=float)
    for i in range(n_qubits):
        zz[i, i] = 1.0  # <Z_i^2> = 1
        for j in range(i + 1, n_qubits):
            op = _pair_operator(n_qubits, i, j)
            zz[i, j] = float(np.real(np.vdot(state, op @ state)))
            zz[j, i] = zz[i, j]
    return zz


def structure_factor_vectorized(
    zz_matrix: np.ndarray,
    n_qubits: int,
    q_grid: np.ndarray,
) -> np.ndarray:
    """Compute S(q) on a q-grid using the vectorized formula.

    S(q) = (1/N) v(q)^† C v(q) where v_j(q) = e^{iqj}, C_ij = <Z_i Z_j>

    This is always >= 0 because it equals (1/N) < |sum_j e^{iqj} Z_j|^2 >.
    """
    sites = np.arange(n_qubits)
    s_q = np.zeros(len(q_grid), dtype=float)
    for qi, q in enumerate(q_grid):
        v = np.exp(1j * q * sites)
        s_q[qi] = np.real(np.vdot(v, zz_matrix @ v)) / n_qubits
    return s_q


def compute_structure_factor_corrected(
    zz_matrix: np.ndarray,
    n_qubits: int,
    n_q_grid: int = 256,
    q_max: float = 2 * np.pi,
) -> dict:
    """Compute structure factor on a dense q-grid and extract key quantities.

    Uses the CORRECT formula S(q) = (1/N) v^† C v (including diagonal).
    Grid spans [0, 2π) to capture q* = π/2 and q* = 3π/2.
    """
    q_grid = np.linspace(0, q_max, n_q_grid, endpoint=False)
    s_q = structure_factor_vectorized(zz_matrix, n_qubits, q_grid)

    q_star = float(q_grid[np.argmax(s_q)])
    s_max = float(np.max(s_q))

    # S at specific points (find nearest grid point)
    def s_at(q_target):
        idx = int(round(q_target / q_max * n_q_grid)) % n_q_grid
        return float(s_q[idx])

    return {
        "q_grid": q_grid,
        "s_q": s_q,
        "q_star": q_star,
        "s_at_0": s_at(0.0),
        "s_at_pi2": s_at(np.pi / 2),
        "s_at_pi": s_at(np.pi),
        "s_at_3pi2": s_at(3 * np.pi / 2),
        "s_max": s_max,
    }


def classify_phase_corrected(
    s0: float,
    s_pi2: float,
    s_pi: float,
    q_star: float,
    x_mean: float = 0.0,
    m: float = 0.0,
    m_s: float = 0.0,
    zz_nn: float = 0.0,
    zz_nnn: float = 0.0,
    gap: float | None = None,
    n_qubits: int = 8,
) -> str:
    """Classify ANNNI phase from observables with CORRECT physics.

    ANNNI phases (Z/Z/X convention):
    - Ferromagnetic: q* ≈ 0, S(0) large, M ≠ 0
    - Antiphase: q* ≈ π/2 (period 4: ↑↑↓↓), S(π/2) large
    - Paramagnetic: <X> high, S(q) ~ 1 (flat), M ~ 0
    - Floating: q* at incommensurate position (not 0 or π/2)
    """
    # Thresholds calibrated for N=8 exact ground states
    s_ferro_threshold = 3.0   # S(0) above this → ferromagnetic
    s_anti_threshold = 1.5    # S(π/2) above this → antiphase
    s_float_threshold = 1.5   # S_max above this for floating
    x_paramag_threshold = 0.85  # <X> above this → paramagnetic
    q_tol = 0.5  # Tolerance for q* near special points

    # Normalize q* to [0, 2π)
    q_star_norm = q_star % (2 * np.pi)

    q_near_0 = min(abs(q_star_norm), abs(q_star_norm - 2 * np.pi)) < q_tol
    q_near_pi2 = abs(q_star_norm - np.pi / 2) < q_tol or abs(q_star_norm - 3 * np.pi / 2) < q_tol
    q_incommensurate = not q_near_0 and not q_near_pi2

    # Paramagnetic: high <X> is the strongest indicator
    if x_mean > x_paramag_threshold:
        return "paramagnetic"

    # Ferromagnetic: S(0) dominant, q* ≈ 0
    if s0 > s_ferro_threshold and (q_near_0 or s0 > max(s_pi2, s_pi)):
        return "ferromagnetic"

    # Antiphase: S(π/2) dominant, q* ≈ π/2
    if s_pi2 > s_anti_threshold and (q_near_pi2 or s_pi2 > max(s0, s_pi)):
        return "antiphase"

    # Floating/incommensurate: q* at non-trivial position with significant S_max
    if q_incommensurate and max(s0, s_pi2, s_pi) > s_float_threshold:
        return "floating_candidate"

    # If S(0) still significantly above 1 (paramagnetic baseline), check ferro
    if s0 > 2.5 and q_near_0:
        return "ferromagnetic"

    # If S(π/2) still significant, check antiphase
    if s_pi2 > 1.2 and q_near_pi2:
        return "antiphase"

    # Default: paramagnetic (no strong order)
    return "paramagnetic"


def classify_phase_from_state_corrected(
    state: np.ndarray,
    n_qubits: int,
) -> dict:
    """Classify phase from a state vector using corrected observables."""
    from observables_compat import _single_site_operator

    # Compute expectations
    z_values = np.zeros(n_qubits)
    x_values = np.zeros(n_qubits)
    for i in range(n_qubits):
        op_z = _single_site_operator(n_qubits, i, "z")
        op_x = _single_site_operator(n_qubits, i, "x")
        z_values[i] = float(np.real(np.vdot(state, op_z @ state)))
        x_values[i] = float(np.real(np.vdot(state, op_x @ state)))

    # ZZ matrix
    zz_matrix = compute_zz_matrix(state, n_qubits)

    # Structure factor
    sf = compute_structure_factor_corrected(zz_matrix, n_qubits)

    # Order parameters
    m = float(np.mean(z_values))
    # Period-4 staggered magnetization for antiphase
    pattern = np.array([1, 1, -1, -1] * (n_qubits // 4 + 1))[:n_qubits]
    m_s = float(np.mean(pattern * z_values))
    zz_nn = float(np.mean([zz_matrix[i, (i + 1) % n_qubits] for i in range(n_qubits)]))
    zz_nnn = float(np.mean([zz_matrix[i, (i + 2) % n_qubits] for i in range(n_qubits)]))

    # Classify
    phase = classify_phase_corrected(
        s0=sf["s_at_0"],
        s_pi2=sf["s_at_pi2"],
        s_pi=sf["s_at_pi"],
        q_star=sf["q_star"],
        x_mean=float(np.mean(x_values)),
        m=m,
        m_s=m_s,
        zz_nn=zz_nn,
        zz_nnn=zz_nnn,
        n_qubits=n_qubits,
    )

    return {
        "phase": phase,
        "magnetization": m,
        "staggered_magnetization": m_s,
        "x_mean": float(np.mean(x_values)),
        "z_mean": m,
        "z_abs_mean": float(np.mean(np.abs(z_values))),
        "zz_nn": zz_nn,
        "zz_nnn": zz_nnn,
        "s0": sf["s_at_0"],
        "s_pi2": sf["s_at_pi2"],
        "s_pi": sf["s_at_pi"],
        "q_star": sf["q_star"],
        "s_max": sf["s_max"],
        "q_grid": sf["q_grid"],
        "s_q": sf["s_q"],
    }


# === UNIT TESTS ===

def run_unit_tests():
    """Run unit tests for S(q) and classifier on analytic states."""
    n = 8
    results = {}

    # Test 1: Ferromagnetic |00000000>
    ferro = np.zeros(2**n, dtype=np.complex128)
    ferro[0] = 1.0
    res_ferro = classify_phase_from_state_corrected(ferro, n)
    results["ferromagnetic"] = {
        "phase": res_ferro["phase"],
        "S(0)": res_ferro["s0"],
        "S(π/2)": res_ferro["s_pi2"],
        "q*": res_ferro["q_star"],
        "M": res_ferro["magnetization"],
        "PASS": res_ferro["phase"] == "ferromagnetic"
                and abs(res_ferro["s0"] - 8.0) < 0.01
                and abs(res_ferro["s_pi2"]) < 0.01,
    }

    # Test 2: Antiphase |00110011>
    antiphase = np.zeros(2**n, dtype=np.complex128)
    antiphase[0b00110011] = 1.0
    res_ap = classify_phase_from_state_corrected(antiphase, n)
    results["antiphase"] = {
        "phase": res_ap["phase"],
        "S(0)": res_ap["s0"],
        "S(π/2)": res_ap["s_pi2"],
        "q*": res_ap["q_star"],
        "PASS": res_ap["phase"] == "antiphase"
                and abs(res_ap["s_pi2"] - 4.0) < 0.01
                and abs(res_ap["s0"]) < 0.01,
    }

    # Test 3: Paramagnetic |+>^8
    plus = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    res_pm = classify_phase_from_state_corrected(plus, n)
    results["paramagnetic"] = {
        "phase": res_pm["phase"],
        "S(0)": res_pm["s0"],
        "S(π/2)": res_pm["s_pi2"],
        "<X>": res_pm["x_mean"],
        "PASS": res_pm["phase"] == "paramagnetic"
                and abs(res_pm["s0"] - 1.0) < 0.01
                and res_pm["x_mean"] > 0.9,
    }

    # Test 4: Sanity — S(q) >= 0 for all test states
    all_nonneg = True
    for name, state in [("ferro", ferro), ("antiphase", antiphase), ("paramag", plus)]:
        zz = compute_zz_matrix(state, n)
        q_grid = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        s_q = structure_factor_vectorized(zz, n, q_grid)
        if np.any(s_q < -1e-10):
            all_nonneg = False
            results[f"{name}_nonneg"] = {"min_S": float(np.min(s_q)), "PASS": False}
    results["S(q) >= 0"] = {"PASS": all_nonneg}

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("STRUCTURE FACTOR & CLASSIFIER UNIT TESTS")
    print("=" * 60)
    results = run_unit_tests()
    all_pass = True
    for name, r in results.items():
        passed = r.get("PASS", False)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{name}: {status}")
        for k, v in r.items():
            if k != "PASS":
                print(f"  {k}: {v}")
        if not passed:
            all_pass = False

    print(f"\n{'=' * 60}")
    print(f"ALL TESTS: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print(f"{'=' * 60}")
