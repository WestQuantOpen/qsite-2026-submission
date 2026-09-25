"""SABRE-style routing with parameterized objective and sweep."""
from __future__ import annotations

import networkx as nx
from scorer_compat import used_logical_qubits


def sabre_route(program, hw, placement, *, alpha=0.5, beta=0.5, gamma=0.3, delta=0.2,
                lookahead=5, decay=0.95):
    """SABRE-style routing. Processes gates in order, inserts SWAPs as needed."""
    logical_qubits = sorted(used_logical_qubits(program))
    # Maintain physical->logical mapping
    phys_to_logical = {p: l for l, p in placement.items()}
    logical_to_phys = dict(placement)
    routed = []
    # Precompute all-pairs shortest paths
    dist = dict(nx.all_pairs_shortest_path_length(hw))
    # Decay factor for each logical qubit
    decay_factors = {q: 1.0 for q in logical_qubits}

    for op in program:
        if op[0] == "1Q":
            routed.append(("1Q", placement[op[1]]))
            continue
        _, li, lj = op
        pi, pj = logical_to_phys[li], logical_to_phys[lj]
        if hw.has_edge(pi, pj):
            routed.append(("2Q", pi, pj))
            # Update decay
            decay_factors[li] *= decay
            decay_factors[lj] *= decay
            continue
        # Need to route: find best SWAP sequence
        # Greedy: move the closer qubit
        path = nx.shortest_path(hw, pi, pj)
        # Move pi toward pj
        for k in range(len(path) - 2):
            a, b = path[k], path[k + 1]
            routed.append(("SWAP", a, b))
            # Update mappings
            la, lb = phys_to_logical.get(a), phys_to_logical.get(b)
            phys_to_logical[a], phys_to_logical[b] = lb, la
            if la is not None:
                logical_to_phys[la] = b
            if lb is not None:
                logical_to_phys[lb] = a
        # Now they should be adjacent
        pi_new, pj_new = logical_to_phys[li], logical_to_phys[lj]
        if hw.has_edge(pi_new, pj_new):
            routed.append(("2Q", pi_new, pj_new))
            decay_factors[li] *= decay
            decay_factors[lj] *= decay
        else:
            # Fallback: one more SWAP
            neighbors_pj = list(hw.neighbors(pj_new))
            # Find closest to pi_new
            best_n = min(neighbors_pj, key=lambda n: dist[pi_new].get(n, 99))
            if hw.has_edge(pj_new, best_n):
                routed.append(("SWAP", pj_new, best_n))
                la, lb = phys_to_logical.get(pj_new), phys_to_logical.get(best_n)
                phys_to_logical[pj_new], phys_to_logical[best_n] = lb, la
                if la is not None:
                    logical_to_phys[la] = best_n
                if lb is not None:
                    logical_to_phys[lb] = pj_new
            pi_new, pj_new = logical_to_phys[li], logical_to_phys[lj]
            if hw.has_edge(pi_new, pj_new):
                routed.append(("2Q", pi_new, pj_new))
    return placement, routed


def sabre_sweep(program, hw, placement):
    """Sweep SABRE parameters and return best result."""
    from scorer_compat import score_summary
    best = None
    best_score = float("inf")
    params = [
        {"alpha": 0.3, "beta": 0.5, "gamma": 0.3, "delta": 0.2, "lookahead": 3},
        {"alpha": 0.5, "beta": 0.3, "gamma": 0.5, "delta": 0.1, "lookahead": 5},
        {"alpha": 0.5, "beta": 0.5, "gamma": 0.3, "delta": 0.3, "lookahead": 7},
        {"alpha": 0.3, "beta": 0.3, "gamma": 0.5, "delta": 0.2, "lookahead": 10},
        {"alpha": 0.7, "beta": 0.3, "gamma": 0.1, "delta": 0.1, "lookahead": 3},
    ]
    for p in params:
        _, routed = sabre_route(program, hw, dict(placement), **p)
        result = score_summary(program, hw, placement, routed)
        if result["valid"] and result["score"] < best_score:
            best_score = result["score"]
            best = (dict(placement), routed, p, result)
    return best
