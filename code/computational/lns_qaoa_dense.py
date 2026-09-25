"""LNS attack on QAOA and continued dense_random optimization."""
from __future__ import annotations

import json, random, time, sys, os
from pathlib import Path

HYBRID_BASE = Path(__file__).resolve().parent.parent
BASE = HYBRID_BASE.parent

def lns_attack(benchmark_name, time_budget=600):
    import sys
    sys.path.insert(0, str(BASE / "common"))
    sys.path.insert(0, str(BASE / "computational" / "classical" / "solvers"))
    sys.path.insert(0, str(HYBRID_BASE / "common"))
    sys.path.insert(0, str(HYBRID_BASE / "computational"))

    from hardware_compat import build_hardware_graph
    from benchmarks_compat import BENCHMARKS
    from scorer_compat import score_summary, used_logical_qubits
    import sabre as sabre_mod
    from edge_meeting_router import edge_meeting_route
    import networkx as nx

    hw = build_hardware_graph()
    program = BENCHMARKS[benchmark_name]
    logical_qubits = sorted(used_logical_qubits(program))
    physical_nodes = sorted(hw.nodes())

    best_path = HYBRID_BASE / "computational" / "best.json"
    with open(best_path) as f:
        best = json.load(f)
    b_best = best["per_benchmark"].get(benchmark_name, {})
    current_score = b_best.get("score", 999)
    current_plc = {int(k): v for k, v in b_best.get("placement", {}).items()}
    print(f"[{benchmark_name}] Current: score={current_score}, swaps={b_best.get('swaps')}, depth={b_best.get('depth')}")

    best_score = current_score
    best_plc = current_plc
    best_strategy = b_best.get("strategy", "lns")
    improvements = 0
    t0 = time.time()

    for iteration in range(10000):
        if time.time() - t0 > time_budget:
            print(f"  Time limit reached at iteration {iteration}")
            break

        # Destroy: reassign 2-6 logical qubits
        plc = best_plc.copy()
        n_destroy = random.randint(2, 6)
        destroy_qubits = random.sample(logical_qubits, min(n_destroy, len(logical_qubits)))
        original_positions = {q: plc[q] for q in destroy_qubits}

        used_physical = set(plc.values())
        available = [p for p in physical_nodes if p not in used_physical or p in original_positions.values()]

        for q in destroy_qubits:
            if available:
                new_pos = random.choice(available)
                available.remove(new_pos)
                plc[q] = new_pos

        # Repair with both routers
        for router_name, router_fn in [("edge_meeting", edge_meeting_route), ("sabre", sabre_mod.sabre_route)]:
            try:
                _, routed = router_fn(program, hw, plc)
                result = score_summary(program, hw, plc, routed)
                if result["valid"] and result["score"] < best_score:
                    best_score = result["score"]
                    best_plc = plc
                    best_strategy = f"lns_{router_name}"
                    improvements += 1
                    print(f"  [LNS IMPROVE] iter {iteration}: {result['score']} (swaps={result['swap_count']}, depth={result['depth']})")
            except Exception:
                pass

    print(f"[{benchmark_name}] LNS complete: {improvements} improvements, best={best_score}")

    if best_score < current_score:
        # Verify and save
        if "edge" in best_strategy:
            _, routed = edge_meeting_route(program, hw, best_plc)
        else:
            _, routed = sabre_mod.sabre_route(program, hw, best_plc)
        result = score_summary(program, hw, best_plc, routed)
        if result["valid"] and result["score"] < current_score:
            valid_benchmarks = ['ghz_star', 'chain_trotter', 'ladder_trotter', 'qaoa_random', 'dense_random', 'vqe_layers']
            best['per_benchmark'] = {k: v for k, v in best['per_benchmark'].items() if k in valid_benchmarks}
            best['per_benchmark'][benchmark_name] = {
                'score': result['score'],
                'swaps': result['swap_count'],
                'depth': result['depth'],
                'placement': {str(k): v for k, v in best_plc.items()},
                'strategy': best_strategy,
            }
            best['total_score'] = sum(v.get('score', 0) for v in best['per_benchmark'].values())
            best['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            tmp_path = best_path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(best, f, indent=2, default=str)
            os.replace(tmp_path, best_path)
            print(f"  Saved! New total: {best['total_score']}")
        else:
            print(f"  Verification FAILED")
    else:
        print(f"  No improvement")


if __name__ == "__main__":
    # QAOA: 15 min
    print("=== LNS on QAOA ===")
    lns_attack("qaoa_random", time_budget=900)
    
    # Dense: 15 min more
    print("\n=== LNS on dense_random (continued) ===")
    lns_attack("dense_random", time_budget=900)
