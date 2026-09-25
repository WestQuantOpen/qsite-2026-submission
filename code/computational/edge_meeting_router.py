"""Edge-Meeting Router — dual-ended movement where both qubits move toward a meeting edge.

Instead of moving one qubit toward the other (SABRE's approach), this router
considers all hardware edges as potential "meeting edges" and moves BOTH qubits
simultaneously toward the meeting point.

Cost: C(e) = N_SWAP + 0.5 * D_movement + λ * C_future
"""
from __future__ import annotations

import networkx as nx
from collections import deque
from typing import Optional

try:
    from scorer_compat import score_summary, used_logical_qubits
except ImportError:
    pass


def edge_meeting_route(
    program: list[tuple],
    hw: nx.Graph,
    placement: dict[int, int],
    lookahead: int = 5,
    parallel_movement: bool = True,
):
    """Route using edge-meeting strategy.

    For each 2Q gate between logical qubits (a, b):
    1. Find all hardware edges that could serve as meeting points
    2. For each meeting edge (u,v), compute cost of moving a->u and b->v (or a->v, b->u)
    3. Choose the meeting edge that minimizes total SWAPs + depth
    4. Execute SWAPs to bring both qubits to the meeting edge in parallel

    Returns: (placement, routed_program)
    """
    logical_qubits = sorted(used_logical_qubits(program))
    phys_to_logical = {p: l for l, p in placement.items()}
    # Initialize all physical qubits
    for p in hw.nodes:
        if p not in phys_to_logical:
            phys_to_logical[p] = None
    logical_to_phys = {l: p for l, p in placement.items()}

    routed = []
    dist = dict(nx.all_pairs_shortest_path_length(hw))
    paths = dict(nx.all_pairs_shortest_path(hw))

    # Track which physical qubits are "busy" (just used) for depth optimization
    last_layer = {}  # phys qubit -> last layer

    for op in program:
        if op[0] == "1Q":
            routed.append(("1Q", placement[op[1]]))
            continue

        _, la, lb = op
        pa, pb = logical_to_phys[la], logical_to_phys[lb]

        if hw.has_edge(pa, pb):
            routed.append(("2Q", pa, pb))
            continue

        # Find best meeting edge
        best_edge = None
        best_cost = float("inf")
        best_path_a = []
        best_path_b = []

        for u, v in hw.edges():
            # Option 1: a -> u, b -> v
            cost_a = dist[pa][u] if pa != u else 0
            cost_b = dist[pb][v] if pb != v else 0
            # Prefer edges where both qubits need to move (parallelism opportunity)
            if parallel_movement:
                # If both need to move, they can move in parallel -> depth = max(cost_a, cost_b)
                depth_cost = max(cost_a, cost_b)
            else:
                depth_cost = cost_a + cost_b
            total_cost = cost_a + cost_b + 0.5 * depth_cost

            # Lookahead: prefer edges close to future gates
            future_cost = _lookahead_cost(la, lb, u, v, program,
                                           program.index(op), logical_to_phys, dist, lookahead)
            total_cost += 0.3 * future_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_edge = (u, v)
                best_path_a = paths[pa][u] if pa != u else [pa]
                best_path_b = paths[pb][v] if pb != v else [pb]

            # Option 2: a -> v, b -> u
            cost_a = dist[pa][v] if pa != v else 0
            cost_b = dist[pb][u] if pb != u else 0
            if parallel_movement:
                depth_cost = max(cost_a, cost_b)
            else:
                depth_cost = cost_a + cost_b
            total_cost = cost_a + cost_b + 0.5 * depth_cost
            future_cost = _lookahead_cost(la, lb, v, u, program,
                                           program.index(op), logical_to_phys, dist, lookahead)
            total_cost += 0.3 * future_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_edge = (v, u)
                best_path_a = paths[pa][v] if pa != v else [pa]
                best_path_b = paths[pb][u] if pb != u else [pb]

        if best_edge is None:
            # Fallback: simple shortest path
            path = paths[pa][pb]
            for k in range(len(path) - 1):
                a, b = path[k], path[k+1]
                routed.append(("SWAP", a, b))
                phys_to_logical[a], phys_to_logical[b] = phys_to_logical[b], phys_to_logical[a]
            logical_to_phys = {l: p for p, l in phys_to_logical.items()}
            routed.append(("2Q", logical_to_phys[la], logical_to_phys[lb]))
            continue

        # Execute parallel SWAPs along both paths
        # Interleave SWAPs from path_a and path_b for parallelism
        swaps_a = []
        for k in range(len(best_path_a) - 1):
            a, b = best_path_a[k], best_path_a[k+1]
            swaps_a.append(("SWAP", a, b))

        swaps_b = []
        for k in range(len(best_path_b) - 1):
            a, b = best_path_b[k], best_path_b[k+1]
            swaps_b.append(("SWAP", a, b))

        # Interleave for parallel execution
        # But check that SWAPs don't conflict (use same physical qubits)
        max_len = max(len(swaps_a), len(swaps_b))
        for i in range(max_len):
            if i < len(swaps_a):
                swap = swaps_a[i]
                routed.append(swap)
                phys_to_logical[swap[1]], phys_to_logical[swap[2]] = \
                    phys_to_logical[swap[2]], phys_to_logical[swap[1]]
            if i < len(swaps_b):
                swap = swaps_b[i]
                # Check no conflict with last SWAP
                if routed and routed[-1][0] == "SWAP":
                    last = routed[-1]
                    if swap[1] in (last[1], last[2]) or swap[2] in (last[1], last[2]):
                        # Conflict: add to next layer (just append, scheduler will handle)
                        pass
                routed.append(swap)
                phys_to_logical[swap[1]], phys_to_logical[swap[2]] = \
                    phys_to_logical[swap[2]], phys_to_logical[swap[1]]

        logical_to_phys = {l: p for p, l in phys_to_logical.items()}

        # Execute the gate
        pa_new = logical_to_phys[la]
        pb_new = logical_to_phys[lb]
        if hw.has_edge(pa_new, pb_new):
            routed.append(("2Q", pa_new, pb_new))
        else:
            # Fallback: should not happen if routing was correct
            # But just in case, do a simple path
            path = paths[pa_new][pb_new]
            for k in range(len(path) - 1):
                a, b = path[k], path[k+1]
                routed.append(("SWAP", a, b))
                phys_to_logical[a], phys_to_logical[b] = phys_to_logical[b], phys_to_logical[a]
            logical_to_phys = {l: p for p, l in phys_to_logical.items()}
            routed.append(("2Q", logical_to_phys[la], logical_to_phys[lb]))

    return placement, routed


def _lookahead_cost(la, lb, meet_u, meet_v, program, op_idx, logical_to_phys, dist, lookahead):
    """Estimate future cost after placing la at meet_u and lb at meet_v."""
    cost = 0
    future_gates = []
    for i in range(op_idx + 1, min(op_idx + 1 + lookahead, len(program))):
        if program[i][0] == "2Q":
            future_gates.append(program[i])

    # Simulate: la is now at meet_u, lb at meet_v
    temp_mapping = dict(logical_to_phys)
    temp_mapping[la] = meet_u
    temp_mapping[lb] = meet_v

    for fg in future_gates[:3]:
        _, fa, fb = fg
        if fa in temp_mapping and fb in temp_mapping:
            pfa, pfb = temp_mapping[fa], temp_mapping[fb]
            if pfa != pfb and not dist.get(pfa, {}).get(pfb, 0) == 0:
                cost += dist[pfa][pfb] - 1 if pfa in dist and pfb in dist[pfa] else 5

    return cost


def edge_meeting_with_swap_prefetch(program, hw, placement, lookahead=8):
    """Edge-meeting router with SWAP prefetch: while a gate runs, start moving
    other qubits toward future interaction partners on disjoint physical qubits.
    """
    logical_to_phys = {l: p for l, p in placement.items()}
    phys_to_logical = {p: l for l, p in placement.items()}
    routed = []
    dist = dict(nx.all_pairs_shortest_path_length(hw))
    paths = dict(nx.all_pairs_shortest_path(hw))

    for i, op in enumerate(program):
        if op[0] == "1Q":
            routed.append(("1Q", placement[op[1]]))
            continue

        _, la, lb = op
        pa, pb = logical_to_phys[la], logical_to_phys[lb]

        if hw.has_edge(pa, pb):
            routed.append(("2Q", pa, pb))
            # SWAP prefetch: while this gate runs, move future qubits
            prefetch_swaps = _compute_prefetch(program, i, hw, phys_to_logical,
                                                logical_to_phys, dist, paths, exclude={pa, pb})
            routed.extend(prefetch_swaps)
            for sw in prefetch_swaps:
                if sw[0] == "SWAP":
                    phys_to_logical[sw[1]], phys_to_logical[sw[2]] = \
                        phys_to_logical[sw[2]], phys_to_logical[sw[1]]
            logical_to_phys = {l: p for p, l in phys_to_logical.items()}
            continue

        # Route using edge-meeting
        best_edge = _find_best_meeting_edge(la, lb, pa, pb, hw, dist, program, i, logical_to_phys, lookahead)
        if best_edge:
            u, v = best_edge
            path_a = paths[pa][u] if pa != u else [pa]
            path_b = paths[pb][v] if pb != v else [pb]

            # Execute interleaved SWAPs
            for k in range(max(len(path_a) - 1, len(path_b) - 1)):
                if k < len(path_a) - 1:
                    a, b = path_a[k], path_a[k+1]
                    routed.append(("SWAP", a, b))
                    phys_to_logical[a], phys_to_logical[b] = phys_to_logical[b], phys_to_logical[a]
                if k < len(path_b) - 1:
                    a, b = path_b[k], path_b[k+1]
                    routed.append(("SWAP", a, b))
                    phys_to_logical[a], phys_to_logical[b] = phys_to_logical[b], phys_to_logical[a]

            logical_to_phys = {l: p for p, l in phys_to_logical.items()}
            routed.append(("2Q", logical_to_phys[la], logical_to_phys[lb]))
        else:
            # Fallback
            path = paths[pa][pb]
            for k in range(len(path) - 1):
                a, b = path[k], path[k+1]
                routed.append(("SWAP", a, b))
                phys_to_logical[a], phys_to_logical[b] = phys_to_logical[b], phys_to_logical[a]
            logical_to_phys = {l: p for p, l in phys_to_logical.items()}
            routed.append(("2Q", logical_to_phys[la], logical_to_phys[lb]))

    return placement, routed


def _find_best_meeting_edge(la, lb, pa, pb, hw, dist, program, op_idx, mapping, lookahead):
    """Find the best meeting edge for the current gate."""
    best_edge = None
    best_cost = float("inf")

    for u, v in hw.edges():
        for (a_target, b_target) in [(u, v), (v, u)]:
            cost_a = dist[pa][a_target] if pa != a_target else 0
            cost_b = dist[pb][b_target] if pb != b_target else 0
            depth_cost = max(cost_a, cost_b)
            total = cost_a + cost_b + 0.5 * depth_cost

            # Simple lookahead
            if op_idx + 1 < len(program) and program[op_idx + 1][0] == "2Q":
                _, fa, fb = program[op_idx + 1]
                temp_map = dict(mapping)
                temp_map[la] = a_target
                temp_map[lb] = b_target
                if fa in temp_map and fb in temp_map:
                    pfa, pfb = temp_map[fa], temp_map[fb]
                    if pfa in dist and pfb in dist[pfa]:
                        total += 0.3 * max(0, dist[pfa][pfb] - 1)

            if total < best_cost:
                best_cost = total
                best_edge = (a_target, b_target)

    return best_edge


def _compute_prefetch(program, gate_idx, hw, phys_to_logical, logical_to_phys, dist, paths, exclude):
    """Compute SWAP prefetches for future gates while current gate runs."""
    prefetch = []
    # Look at next 2-3 gates
    for i in range(gate_idx + 1, min(gate_idx + 4, len(program))):
        if program[i][0] != "2Q":
            continue
        _, la, lb = program[i]
        if la not in logical_to_phys or lb not in logical_to_phys:
            continue
        pa, pb = logical_to_phys[la], logical_to_phys[lb]
        if hw.has_edge(pa, pb):
            continue  # Already adjacent

        # Move the closer qubit by one step, if it doesn't conflict
        path = paths.get(pa, {}).get(pb, [pa, pb])
        if len(path) >= 3:
            a, b = path[0], path[1]
            if a not in exclude and b not in exclude:
                prefetch.append(("SWAP", a, b))
                exclude.add(a)
                exclude.add(b)
                break

    return prefetch


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "computational" / "classical" / "solvers"))
    from hardware_compat import build_hardware_graph
    from benchmarks_compat import BENCHMARKS
    from scorer_compat import score_summary
    import placement as placement_mod

    hw = build_hardware_graph()
    for name in ["dense_random", "qaoa_random", "ghz_star"]:
        program = BENCHMARKS[name]
        plc = placement_mod.spectral_placement(program, hw)
        _, routed = edge_meeting_route(program, hw, plc)
        result = score_summary(program, hw, plc, routed)
        print(f"{name}: score={result['score']:.1f}, swaps={result['swap_count']}, depth={result['depth']}, valid={result['valid']}")

        _, routed2 = edge_meeting_with_swap_prefetch(program, hw, plc)
        result2 = score_summary(program, hw, plc, routed2)
        print(f"  +prefetch: score={result2['score']:.1f}, swaps={result2['swap_count']}, depth={result2['depth']}")
