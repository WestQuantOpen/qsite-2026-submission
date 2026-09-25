"""Placement optimizers: degree matching, spectral, SA, multi-start, hill climbing."""
from __future__ import annotations

import random
import networkx as nx
import numpy as np
from scorer_compat import used_logical_qubits


def build_weighted_interaction_graph(program):
    g = nx.Graph()
    for op in program:
        if op[0] == "2Q":
            i, j = op[1], op[2]
            if g.has_edge(i, j):
                g[i][j]["weight"] += 1
            else:
                g.add_edge(i, j, weight=1)
    return g


def placement_cost(placement, wig, hw):
    """C_place = sum w_ij * d(P(i), P(j))"""
    cost = 0
    for i, j, d in wig.edges(data=True):
        pi, pj = placement.get(i), placement.get(j)
        if pi is not None and pj is not None:
            try:
                cost += d.get("weight", 1) * nx.shortest_path_length(hw, pi, pj)
            except nx.NetworkXNoPath:
                cost += d.get("weight", 1) * 20
    return cost


def degree_matching(program, hw):
    """Match logical degrees to physical degrees."""
    logical_qubits = sorted(used_logical_qubits(program))
    ig = build_weighted_interaction_graph(program)
    hw_degrees = sorted(hw.nodes(), key=lambda x: -dict(hw.degree())[x])
    logical_degrees = sorted(logical_qubits, key=lambda x: -dict(ig.degree(x, weight="weight")).get(x, 0))
    return {logical_degrees[i]: hw_degrees[i] for i in range(len(logical_qubits))}


def spectral_placement(program, hw):
    """Spectral ordering placement."""
    logical_qubits = sorted(used_logical_qubits(program))
    ig = build_weighted_interaction_graph(program)
    # Get spectral layout of interaction graph
    try:
        pos_ig = nx.spectral_layout(ig)
        pos_hw = nx.spectral_layout(hw)
        # Sort by first coordinate
        logical_sorted = sorted(logical_qubits, key=lambda x: pos_ig.get(x, (0, 0))[0])
        physical_sorted = sorted(hw.nodes(), key=lambda x: pos_hw.get(x, (0, 0))[0])
        return {logical_sorted[i]: physical_sorted[i] for i in range(len(logical_qubits))}
    except Exception:
        return {logical_qubits[i]: list(hw.nodes())[i] for i in range(len(logical_qubits))}


def random_placement(program, hw, seed=None):
    """Random placement."""
    rng = random.Random(seed)
    logical_qubits = sorted(used_logical_qubits(program))
    physical = list(hw.nodes())
    rng.shuffle(physical)
    return {logical_qubits[i]: physical[i] for i in range(len(logical_qubits))}


def simulated_annealing_placement(program, hw, initial=None, iterations=500, seed=None):
    """Simulated annealing on placement."""
    rng = random.Random(seed)
    logical_qubits = sorted(used_logical_qubits(program))
    wig = build_weighted_interaction_graph(program)
    if initial is None:
        placement = random_placement(program, hw, seed=rng.random())
    else:
        placement = dict(initial)
    current_cost = placement_cost(placement, wig, hw)
    best = dict(placement)
    best_cost = current_cost
    physical_nodes = list(hw.nodes())
    for it in range(iterations):
        temp = max(0.01, 1.0 - it / iterations)
        # Swap two physical assignments
        q1, q2 = rng.sample(logical_qubits, 2)
        new_placement = dict(placement)
        new_placement[q1], new_placement[q2] = placement[q2], placement[q1]
        new_cost = placement_cost(new_placement, wig, hw)
        if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / temp):
            placement = new_placement
            current_cost = new_cost
            if current_cost < best_cost:
                best = dict(placement)
                best_cost = current_cost
    return best


def multi_start_optimize(program, hw, n_starts=20, sa_iters=300, seed=None):
    """Multi-start placement optimization."""
    rng = random.Random(seed)
    wig = build_weighted_interaction_graph(program)
    best = None
    best_cost = float("inf")
    # Include degree matching and spectral as starts
    starts = [degree_matching(program, hw), spectral_placement(program, hw)]
    for _ in range(n_starts - 2):
        starts.append(random_placement(program, hw, seed=rng.random()))
    for start in starts:
        refined = simulated_annealing_placement(program, hw, initial=start, iterations=sa_iters, seed=rng.random())
        cost = placement_cost(refined, wig, hw)
        if cost < best_cost:
            best = refined
            best_cost = cost
    return best


def hill_climb_placement(program, hw, initial=None, max_iters=200):
    """Hill climbing on placement (swap pairs, keep if better)."""
    logical_qubits = sorted(used_logical_qubits(program))
    wig = build_weighted_interaction_graph(program)
    if initial is None:
        placement = random_placement(program, hw)
    else:
        placement = dict(initial)
    current_cost = placement_cost(placement, wig, hw)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(logical_qubits)):
            for j in range(i + 1, len(logical_qubits)):
                q1, q2 = logical_qubits[i], logical_qubits[j]
                new_placement = dict(placement)
                new_placement[q1], new_placement[q2] = placement[q2], placement[q1]
                new_cost = placement_cost(new_placement, wig, hw)
                if new_cost < current_cost:
                    placement = new_placement
                    current_cost = new_cost
                    improved = True
    return placement
