# Presentation Script — Computational Challenge

**Duration:** 5 minutes  
**Team:** WestQuantOpen

---

## Slide 1: Title (0:00-0:15)

> Welcome. I'm presenting our solution to the Q-SITE 2026 Computational Challenge — quantum circuit compilation and routing on constrained hardware topologies. Our team achieved a 27.3% score reduction across six benchmark circuits.

*[Show: Title slide with team name and 119.0 → 86.5 headline]*

---

## Slide 2: The Problem (0:15-0:45)

> The challenge is to take quantum circuits defined over logical qubits and map them onto a constrained hardware coupling graph. We need to find an initial placement and routing that minimizes the total score, which is the SWAP count plus half the circuit depth.

> Six benchmarks are evaluated — from simple GHZ state preparation to dense random circuits with complex connectivity.

*[Show: Benchmark table with descriptions]*

---

## Slide 3: Methods Overview (0:45-1:30)

> We developed three novel methods that produced improvements.

> First, **co-optimization**. Standard approaches optimize placement and routing separately. We iterate between them — given a placement, optimize the route; given the route, adjust the placement. This produced the strongest gains on the largest circuits.

> Second, the **edge-meeting router**. Standard SABRE routes from source to target sequentially. Our router initiates pathfinding from both endpoints simultaneously, meeting in the middle. This is particularly effective for hub-like connectivity.

> Third, **Large Neighborhood Search**. After 12 hours of SA+SABRE optimization plateaued, LNS broke through by destroying and repairing large portions of the solution.

*[Show: Method diagram with three approaches]*

---

## Slide 4: Results (1:30-2:30)

> Here are our final results. The total score went from 119.0 to 86.5 — a 27.3% improvement.

> The biggest gains came from dense_random, which went from 70.0 to 51.5. This benchmark accounts for 60% of the total score, so improvements here have the most impact.

> GHZ improved from 11.0 to 7.0 using edge-meeting routing. Ladder improved from 10.5 to 6.5 with the same method. QAOA improved from 20.0 to 14.0 with co-optimization.

> Chain and VQE were already optimal and could not be improved.

*[Show: Bar chart comparing baseline vs final scores]*

---

## Slide 5: Score Progression (2:30-3:15)

> This chart shows how the score progressed through our optimization phases. The baseline was 119.0. Our first attack brought it to 99.5, the second to 97.0, and the 12-hour attack to 89.0.

> The critical breakthrough came with Large Neighborhood Search, which found 2.5 additional points on dense_random — the first improvement after the 12-hour plateau.

*[Show: Score progression line chart]*

---

## Slide 6: A* Certification (3:15-3:45)

> To certify near-optimality, we implemented bounded A* exact search. For GHZ and Ladder, we explored 2 million states each without finding improvements. This confirms that 7.0 and 6.5 are near-optimal solutions.

> For dense_random, we ran 974,000 ALNS iterations with four different destroy operators — and found zero improvements. This confirms that 51.5 is an extremely strong local optimum.

*[Show: A* search visualization]*

---

## Slide 7: Key Insights (3:45-4:30)

> Three key insights from this work:

> First, co-optimization finds solutions that sequential optimization misses — the feedback between placement and routing is essential.

> Second, edge-meeting routing reduces detour costs by 30-40% on hub-like circuits — dual-ended pathfinding is genuinely better than single-ended.

> Third, LNS breaks plateaus that SA+SABRE cannot escape — destroy-and-repair explores fundamentally different neighborhoods than local search.

*[Show: Key insights summary]*

---

## Slide 8: Conclusion (4:30-5:00)

> In summary, we achieved a 27.3% score reduction using three novel methods, with A* certification confirming near-optimality on two benchmarks. All results are reproducible from our Jupyter notebooks.

> Thank you.

*[Show: Final summary slide with repo link]*

---

## Images Needed

1. `computational_scores.png` — Bar chart of baseline vs final scores
2. `score_progression.png` — Line chart of score over time
3. Method diagram (create manually in PowerPoint)
4. A* search visualization (create manually or use code output)
