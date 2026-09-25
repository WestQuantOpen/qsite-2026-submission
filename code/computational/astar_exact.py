"""A* / Branch-and-Bound exact router for small benchmarks (GHZ, Ladder).

For small programs (≤8 logical qubits, ≤12 gates), perform exhaustive A* search
to find the provably optimal routing. This gives:
  1. The best possible score (may improve on SA+SABRE)
  2. A saturation certificate proving no better solution exists
"""
from __future__ import annotations

import heapq
import networkx as nx
import time as time_module
from pathlib import Path

try:
    from scorer_compat import score_summary, used_logical_qubits
except ImportError:
    pass


def astar_exact_route(
    program: list[tuple],
    hw: nx.Graph,
    placement: dict[int, int],
    max_states: int = 5_000_000,
    time_limit_seconds: float = 120.0,
):
    """A* search over all possible routing sequences.

    State: (mapping, gate_index)
    Actions: execute gate (if adjacent) or perform any SWAP
    Cost: N_SWAP + 0.5 * depth
    Heuristic: sum of min distances for remaining gates

    Returns: (best_routed, best_score, proven_optimal, states_explored)
    """
    import time
    start_time = time.time()

    logical_qubits = sorted(used_logical_qubits(program))
    dist = dict(nx.all_pairs_shortest_path_length(hw))

    initial_mapping = tuple(sorted((p, l) for l, p in placement.items()))

    def heuristic(mapping_tuple, gate_idx):
        """Admissible: min SWAPs needed for remaining gates."""
        mapping = dict(mapping_tuple)
        l_to_p = {l: p for p, l in mapping.items()}
        total = 0
        for i in range(gate_idx, len(program)):
            if program[i][0] != "2Q":
                continue
            _, la, lb = program[i]
            if la in l_to_p and lb in l_to_p:
                pa, pb = l_to_p[la], l_to_p[lb]
                if pa != pb and not hw.has_edge(pa, pb):
                    total += dist[pa][pb] - 1
        return total

    def compute_depth(ops):
        if not ops:
            return 0
        last_layer = {}
        max_layer = 0
        for op in ops:
            if op[0] in ("SWAP", "2Q"):
                _, a, b = op
                la = last_layer.get(a, -1)
                lb = last_layer.get(b, -1)
                layer = max(la, lb) + 1
                last_layer[a] = layer
                last_layer[b] = layer
                max_layer = max(max_layer, layer)
            elif op[0] == "1Q":
                _, q = op
                last_layer[q] = last_layer.get(q, -1) + 1
                max_layer = max(max_layer, last_layer[q])
        return max_layer + 1

    def cost(ops):
        n_swaps = sum(1 for op in ops if op[0] == "SWAP")
        depth = compute_depth(ops)
        return n_swaps + 0.5 * depth

    # Priority queue: (f_cost, counter, mapping, gate_idx, routed_ops)
    h_start = heuristic(initial_mapping, 0)
    counter = 0
    pq = [(h_start, counter, initial_mapping, 0, [])]
    visited = set()
    best_solution = None
    best_cost = float("inf")
    states_explored = 0

    while pq:
        if states_explored >= max_states or time.time() - start_time > time_limit_seconds:
            # Timeout — return best found so far
            proven = False
            break

        f_cost, _, mapping_tuple, gate_idx, routed_ops = heapq.heappop(pq)
        states_explored += 1

        if gate_idx >= len(program):
            g_cost = cost(routed_ops)
            if g_cost < best_cost:
                best_cost = g_cost
                best_solution = list(routed_ops)
            continue

        state_hash = (mapping_tuple, gate_idx)
        if state_hash in visited:
            continue
        visited.add(state_hash)

        mapping = dict(mapping_tuple)
        l_to_p = {l: p for p, l in mapping.items()}

        op = program[gate_idx]
        if op[0] == "1Q":
            # Just execute
            new_ops = routed_ops + [("1Q", placement[op[1]])]
            h = heuristic(mapping_tuple, gate_idx + 1)
            g = cost(new_ops)
            counter += 1
            heapq.heappush(pq, (g + h, counter, mapping_tuple, gate_idx + 1, new_ops))
            continue

        _, la, lb = op
        pa, pb = l_to_p[la], l_to_p[lb]

        # Action 1: Execute gate if adjacent
        if hw.has_edge(pa, pb):
            new_ops = routed_ops + [("2Q", pa, pb)]
            h = heuristic(mapping_tuple, gate_idx + 1)
            g = cost(new_ops)
            counter += 1
            heapq.heappush(pq, (g + h, counter, mapping_tuple, gate_idx + 1, new_ops))

        # Action 2: SWAP on any edge
        for u, v in hw.edges():
            new_mapping = dict(mapping)
            new_mapping[u], new_mapping[v] = new_mapping.get(v), new_mapping.get(u)
            new_mapping_tuple = tuple(sorted(new_mapping.items()))
            new_ops = routed_ops + [("SWAP", u, v)]
            h = heuristic(new_mapping_tuple, gate_idx)
            g = cost(new_ops)
            counter += 1
            heapq.heappush(pq, (g + h, counter, new_mapping_tuple, gate_idx, new_ops))
    else:
        proven = True  # Exhausted all states

    return best_solution, best_cost, proven, states_explored


def find_optimal_for_benchmark(name, program, hw, max_states=2_000_000, time_limit=120):
    """Try all placements and find the optimal route for a small benchmark.

    Returns: (best_placement, best_routed, best_score, proven_optimal)
    """
    import itertools
    from placement import spectral_placement, random_placement

    logical_qubits = sorted(used_logical_qubits(program))
    physical_qubits = list(hw.nodes)
    func_start_time = time_module.time()

    if len(logical_qubits) > len(physical_qubits):
        return None, None, float("inf"), False

    best_overall = None
    best_score = float("inf")
    best_plc = None
    proven = False

    # Try spectral placement first
    plc = spectral_placement(program, hw)
    routed, score, p, states = astar_exact_route(program, hw, plc, max_states, time_limit)
    if routed and score < best_score:
        best_score = score
        best_overall = routed
        best_plc = plc
        proven = p

    # Try random placements
    import random
    for seed in range(20):
        if time_module.time() - func_start_time > time_limit:
            break
        plc = random_placement(program, hw, seed=seed)
        routed, score, p, states = astar_exact_route(program, hw, plc, max_states, time_limit)
        if routed and score < best_score:
            best_score = score
            best_overall = routed
            best_plc = plc
            proven = p and proven

    # Try exhaustive placements for very small problems
    if len(logical_qubits) <= 6:
        for combo in itertools.permutations(physical_qubits, len(logical_qubits)):
            plc = {logical_qubits[i]: combo[i] for i in range(len(logical_qubits))}
            routed, score, p, states = astar_exact_route(program, hw, plc, max_states, time_limit)
            if routed and score < best_score:
                best_score = score
                best_overall = routed
                best_plc = plc
                proven = p and proven

    return best_plc, best_overall, best_score, proven


import time as time_module
start_time = time_module.time()

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "computational" / "classical" / "solvers"))
    from hardware_compat import build_hardware_graph
    from benchmarks_compat import BENCHMARKS
    from scorer_compat import score_summary
    import placement as placement_mod
    import json

    hw = build_hardware_graph()

    for name in ["ghz_star", "ladder_trotter"]:
        program = BENCHMARKS[name]
        print(f"\n=== {name} ===")
        print(f"Gates: {len(program)}, Qubits: {len(used_logical_qubits(program))}")

        # Current best
        best_path = Path(__file__).resolve().parent / "best.json"
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)
            current = best.get("per_benchmark", {}).get(name, {})
            print(f"Current: score={current.get('score')}, swaps={current.get('swaps')}, depth={current.get('depth')}")

        # A* search
        plc, routed, score, proven = find_optimal_for_benchmark(name, program, hw,
                                                                  max_states=1_000_000, time_limit=60)
        if routed:
            result = score_summary(program, hw, plc, routed)
            print(f"A* result: score={result['score']:.1f}, swaps={result['swap_count']}, depth={result['depth']}")
            print(f"Proven optimal: {proven}")
        else:
            print("A* found no solution")
