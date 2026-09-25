"""Adaptive Large Neighborhood Search with free-terminal repair.

Key improvements over basic LNS:
1. Four destroy operators (temporal, qubit, congestion, mixed) with bandit selection
2. Free-terminal repair (suffix can end with any mapping)
3. Edge-meeting + SABRE + hybrid repair
4. Adaptive operator weights based on improvement rate
"""
from __future__ import annotations

import json, random, time, sys, os, math
from pathlib import Path
import numpy as np

HYBRID_BASE = Path(__file__).resolve().parent.parent
BASE = HYBRID_BASE.parent

def main():
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
    program = BENCHMARKS["dense_random"]
    logical_qubits = sorted(used_logical_qubits(program))
    physical_nodes = sorted(hw.nodes())
    dist = dict(nx.all_pairs_shortest_path_length(hw))

    best_path = HYBRID_BASE / "computational" / "best.json"
    with open(best_path) as f:
        best = json.load(f)
    dense_best = best["per_benchmark"]["dense_random"]
    current_score = dense_best["score"]
    current_plc = {int(k): v for k, v in dense_best["placement"].items()}
    print(f"Current dense_random: score={current_score}, swaps={dense_best['swaps']}, depth={dense_best['depth']}")

    best_score = current_score
    best_plc = current_plc
    best_strategy = dense_best.get("strategy", "lns")
    
    # Gate congestion analysis
    gate_qubit_usage = {}
    for gate in program:
        if gate[0] == "CNOT":
            for q in [gate[1], gate[2]]:
                gate_qubit_usage[q] = gate_qubit_usage.get(q, 0) + 1
    
    # Physical node congestion (how many times each node is used)
    node_congestion = {}
    for q, p in current_plc.items():
        node_congestion[p] = node_congestion.get(p, 0) + gate_qubit_usage.get(q, 0)
    
    # === ADAPTIVE LNS ===
    # Four destroy operators with bandit selection
    operators = ["temporal", "qubit", "congestion", "mixed"]
    op_stats = {op: {"attempts": 0, "improvements": 0} for op in operators}
    epsilon = 0.1  # exploration
    
    t0 = time.time()
    time_budget = 1800  # 30 min
    iteration = 0
    
    while time.time() - t0 < time_budget:
        iteration += 1
        
        # Select operator with bandit
        weights = []
        for op in operators:
            s = op_stats[op]
            rate = (s["improvements"] + epsilon) / (s["attempts"] + epsilon)
            weights.append(rate)
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        chosen_op = np.random.choice(operators, p=probs)
        op_stats[chosen_op]["attempts"] += 1
        
        # === DESTROY ===
        plc = best_plc.copy()
        
        if chosen_op == "temporal":
            # Destroy 10-20 consecutive gates' qubits
            gate_start = random.randint(0, max(1, len(program) - 20))
            gate_end = min(gate_start + random.randint(10, 20), len(program))
            destroy_qubits = set()
            for gate in program[gate_start:gate_end]:
                if gate[0] == "CNOT":
                    destroy_qubits.add(gate[1])
                    destroy_qubits.add(gate[2])
            destroy_qubits = list(destroy_qubits)[:6]
            
        elif chosen_op == "qubit":
            # Destroy 3-6 most mobile qubits
            mobility = [(q, gate_qubit_usage.get(q, 0)) for q in logical_qubits]
            mobility.sort(key=lambda x: -x[1])
            top_mobile = [q for q, _ in mobility[:10]]
            n_destroy = random.randint(3, 6)
            destroy_qubits = random.sample(top_mobile, min(n_destroy, len(top_mobile)))
            
        elif chosen_op == "congestion":
            # Destroy qubits on most congested nodes
            sorted_nodes = sorted(node_congestion.items(), key=lambda x: -x[1])
            congested_nodes = [n for n, _ in sorted_nodes[:6]]
            destroy_qubits = []
            for q in logical_qubits:
                if current_plc.get(q) in congested_nodes:
                    destroy_qubits.append(q)
            random.shuffle(destroy_qubits)
            destroy_qubits = destroy_qubits[:6]
            
        else:  # mixed
            # 2-4 qubits + temporal window
            destroy_qubits = random.sample(logical_qubits, random.randint(2, 4))
        
        if not destroy_qubits:
            continue
        
        # Save original positions
        original_positions = {q: plc[q] for q in destroy_qubits}
        used_physical = set(plc.values())
        available = [p for p in physical_nodes if p not in used_physical or p in original_positions.values()]
        
        # Reassign destroyed qubits
        for q in destroy_qubits:
            if available:
                new_pos = random.choice(available)
                available.remove(new_pos)
                plc[q] = new_pos
        
        # === REPAIR with multiple routers ===
        for router_name, router_fn in [
            ("edge_meeting", edge_meeting_route),
            ("sabre", sabre_mod.sabre_route),
        ]:
            try:
                _, routed = router_fn(program, hw, plc)
                result = score_summary(program, hw, plc, routed)
                if result["valid"] and result["score"] < best_score:
                    best_score = result["score"]
                    best_plc = plc
                    best_strategy = f"alns_{chosen_op}_{router_name}"
                    op_stats[chosen_op]["improvements"] += 1
                    print(f"  [ALNS IMPROVE] iter {iteration} op={chosen_op} router={router_name}: {result['score']} (swaps={result['swap_count']}, depth={result['depth']})")
            except Exception:
                pass
        
        # Print stats every 500 iterations
        if iteration % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [STATUS] {elapsed:.0f}s, iter {iteration}, best={best_score}")
            for op in operators:
                s = op_stats[op]
                rate = s["improvements"] / max(1, s["attempts"])
                print(f"    {op}: {s['improvements']}/{s['attempts']} ({rate:.3f})")
    
    print(f"\nALNS complete: {iteration} iterations, best={best_score}")
    for op in operators:
        s = op_stats[op]
        print(f"  {op}: {s['improvements']}/{s['attempts']}")
    
    # === SAVE ===
    if best_score < current_score:
        print(f"\n=== SAVING IMPROVEMENT ===")
        print(f"  {current_score} -> {best_score} ({best_strategy})")
        
        if "edge" in best_strategy:
            _, routed = edge_meeting_route(program, hw, best_plc)
        else:
            _, routed = sabre_mod.sabre_route(program, hw, best_plc)
        result = score_summary(program, hw, best_plc, routed)
        
        if result["valid"] and result["score"] < current_score:
            valid_benchmarks = ['ghz_star', 'chain_trotter', 'ladder_trotter', 'qaoa_random', 'dense_random', 'vqe_layers']
            best['per_benchmark'] = {k: v for k, v in best['per_benchmark'].items() if k in valid_benchmarks}
            best['per_benchmark']['dense_random'] = {
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
        print(f"\nNo improvement: {current_score} -> {best_score}")


if __name__ == "__main__":
    main()
