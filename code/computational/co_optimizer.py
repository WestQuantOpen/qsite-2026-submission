"""Placement-routing co-optimization for the hybrid search.

SA + evolutionary mutations on placement, evaluated with actual routed score.
"""
from __future__ import annotations

import random
import networkx as nx
import numpy as np


def co_optimize(program, hw, initial_placement=None, iterations=1000, seed=42):
    """Placement-routing co-optimization using SA + mutations.

    Final fitness = actual routed score (not proxy).
    """
    rng = random.Random(seed)
    from scorer_compat import score_summary, used_logical_qubits
    from baseline_routing_compat import solve as baseline_solve
    import sabre as sabre_mod

    def _route(program, hw, plc):
        """Route with SABRE (accepts placement)."""
        _, routed = sabre_mod.sabre_route(program, hw, plc)
        return routed

    logical_qubits = sorted(used_logical_qubits(program))

    # Initialize
    if initial_placement:
        plc = dict(initial_placement)
    else:
        physical = list(hw.nodes())
        rng.shuffle(physical)
        plc = {logical_qubits[i]: physical[i] for i in range(len(logical_qubits))}

    # Evaluate initial
    routed = _route(program, hw, plc)
    result = score_summary(program, hw, plc, routed)
    best_plc = dict(plc)
    best_score = result["score"] if result["valid"] else float("inf")

    current_plc = dict(plc)
    current_score = best_score

    for it in range(iterations):
        temp = max(0.01, 1.0 - it / iterations)

        # Mutation: swap two physical assignments
        q1, q2 = rng.sample(logical_qubits, 2)
        new_plc = dict(current_plc)
        new_plc[q1], new_plc[q2] = current_plc[q2], current_plc[q1]

        # Evaluate with actual router
        new_routed = _route(program, hw, new_plc)
        new_result = score_summary(program, hw, new_plc, new_routed)
        new_score = new_result["score"] if new_result["valid"] else float("inf")

        # Accept?
        if new_score < current_score or rng.random() < np.exp(-(new_score - current_score) / temp):
            current_plc = new_plc
            current_score = new_score
            if current_score < best_score:
                best_plc = dict(current_plc)
                best_score = current_score

    return best_plc, best_score


def evolutionary_co_optimize(program, hw, population_size=20, generations=50, seed=42):
    """Evolutionary placement-routing co-optimization."""
    rng = random.Random(seed)
    from scorer_compat import score_summary, used_logical_qubits
    from baseline_routing_compat import solve as baseline_solve
    import sabre as sabre_mod

    def _route(program, hw, plc):
        """Route with SABRE (accepts placement)."""
        _, routed = sabre_mod.sabre_route(program, hw, plc)
        return routed

    logical_qubits = sorted(used_logical_qubits(program))
    physical = list(hw.nodes())

    # Initialize population
    population = []
    for _ in range(population_size):
        rng.shuffle(physical)
        plc = {logical_qubits[i]: physical[i] for i in range(len(logical_qubits))}
        routed = _route(program, hw, plc)
        result = score_summary(program, hw, plc, routed)
        score = result["score"] if result["valid"] else float("inf")
        population.append((score, plc))

    population.sort(key=lambda x: x[0])
    best_plc = dict(population[0][1])
    best_score = population[0][0]

    for gen in range(generations):
        # Selection: keep top half
        survivors = population[:population_size // 2]

        # Crossover + mutation
        children = []
        for _ in range(population_size - len(survivors)):
            p1, p2 = rng.sample(survivors, 2)
            child = _crossover(p1[1], p2[1], logical_qubits, rng)
            # Mutate
            if rng.random() < 0.3:
                q1, q2 = rng.sample(logical_qubits, 2)
                child[q1], child[q2] = child[q2], child[q1]
            routed = _route(program, hw, child)
            result = score_summary(program, hw, child, routed)
            score = result["score"] if result["valid"] else float("inf")
            children.append((score, child))

        population = survivors + children
        population.sort(key=lambda x: x[0])

        if population[0][0] < best_score:
            best_plc = dict(population[0][1])
            best_score = population[0][0]

    return best_plc, best_score


def _crossover(p1, p2, logical_qubits, rng):
    """Order crossover for placements."""
    child = {}
    used = set()
    # Take first half from p1
    half = len(logical_qubits) // 2
    for q in logical_qubits[:half]:
        child[q] = p1[q]
        used.add(p1[q])
    # Fill rest from p2
    remaining = [p2[q] for q in logical_qubits if p2[q] not in used]
    idx = 0
    for q in logical_qubits[half:]:
        child[q] = remaining[idx]
        idx += 1
    return child
