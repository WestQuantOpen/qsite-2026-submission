# Presentation Script — Scientific Challenge

**Duration:** 7 minutes  
**Team:** WestQuantOpen

---

## Slide 1: Title (0:00-0:15)

> Welcome. I'm presenting our solution to the Q-SITE 2026 Scientific Challenge — quantum phase detection for the ANNNI model using variational quantum eigensolvers. Our team discovered and fixed three critical bugs, built a challenge-compliant noisy pipeline, and developed a novel phase-adapted VQE that solves all phases.

*[Show: Title slide with team name and key results headline]*

---

## Slide 2: The ANNNI Model (0:15-0:45)

> The ANNNI model is a 1D quantum spin chain with nearest-neighbor and next-nearest-neighbor interactions, plus a transverse field. The competition between these interactions creates a rich phase diagram with four phases: ferromagnetic, antiphase, paramagnetic, and floating.

> The key challenge is detecting these phases using quantum circuits and understanding how gate noise affects detection accuracy.

*[Show: ANNNI Hamiltonian and phase diagram schematic]*

---

## Slide 3: Critical Fix 1 — Structure Factor (0:45-1:45)

> Our first discovery was a critical bug in the structure factor computation. The reference implementation was missing the diagonal terms — the i equals j contributions. This produced S(q) minus 1 instead of S(q), causing impossible negative values.

> The correct formula is S(q) equals one over N times the sum over all i,j of e to the i q (i-j) times the Z-Z correlation. This equals one over N times the expectation of the magnitude squared of the sum of e to the i q j Z_j — which is guaranteed to be non-negative.

> We verified this on three analytic states: ferromagnetic gives S(0) equals 8, antiphase gives S(pi/2) equals 4, and paramagnetic gives S(q) approximately 1 for all q.

*[Show: Structure factor plots for three analytic states]*

---

## Slide 4: Critical Fix 2 — Antiphase Wavevector (1:45-2:30)

> Our second discovery was that the classifier was checking S(pi) for antiphase. But ANNNI antiphase has the pattern up-up-down-down, which has period 4. The characteristic ordering wavevector is therefore q* equals pi over 2, not pi.

> After this correction, antiphase classification accuracy went from 0% to 93.3%. Overall classifier accuracy improved from 25.85% to 70.1%, with 100% accuracy on well-separated interior points.

*[Show: Classifier accuracy before vs after bar chart]*

---

## Slide 5: Critical Fix 3 — Phase-Adapted VQE (2:30-4:00)

> Our third and most significant discovery was about VQE. Standard VQE uses U_HVA applied to the zero state. This works perfectly for ferromagnetic — delta E equals 0.003. But it fails completely for antiphase — delta E equals 2.16.

> We ran a simple sanity test: we computed the energy of bare product states at phase-specific points. The antiphase reference state, zero-zero-one-one-zero-zero-one-one, has delta E of only 0.24 at the antiphase point. This means the HVA ansatz CAN represent the state — the problem is state preparation, not ansatz expressivity.

> Our solution: phase-adapted reference-state HVA. Instead of U_HVA applied to zero, we use U_HVA applied to a phase-specific reference state. For ferromagnetic, we use all-zeros. For antiphase, we use zero-zero-one-one. For paramagnetic, we use plus states.

> The result was dramatic. Antiphase delta E went from 2.16 to 0.02. All phases now work with delta E below 0.12.

> Even better, the winning branch — the reference state that produces the lowest energy — provides an independent phase signal. At antiphase points, the antiphase branch wins. At paramagnetic points, the paramag branch wins. This gives us two independent phase detection signals.

*[Show: VQE comparison bar chart (standard vs phase-adapted, log scale)]*

---

## Slide 6: Challenge-Compliant Noisy Pipeline (4:00-5:30)

> With a working VQE, we built a challenge-compliant noisy pipeline. After each HVA layer, we apply a depolarizing channel — the standard noise model for quantum gates. We optimize at zero noise, freeze the parameters, then evaluate at p equals 0.01 and 0.05.

> The key question: which type of quantum order is most noise-sensitive?

> Our results: ferromagnetic order is most sensitive — S(0) degrades by 29.7% at p=0.05. Antiphase is next — S(pi/2) degrades by 25.2%. Paramagnetic is most robust — X expectation degrades by only 18.8%.

> Critically, all phases remain correctly classified at p=0.05. The phase labels are stable under noise, even though the order parameters degrade significantly.

*[Show: Noise degradation plot with three phases]*

---

## Slide 7: Research Narrative (5:30-6:15)

> Our project evolved from "map the ANNNI phase diagram" to a much richer research question: how do representation choice and gate noise jointly affect quantum phase detection?

> The key discoveries were:
> 1. The structure factor implementation had a systematic offset
> 2. The antiphase wavevector is pi/2, not pi
> 3. Shallow HVA works in the ferro region but fails in frustrated regimes without phase-adapted initialization
> 4. Phase-adapted reference states solve all phases
> 5. Ferromagnetic order is most noise-sensitive
> 6. The winning variational branch provides an independent phase signal

*[Show: Research narrative summary]*

---

## Slide 8: Conclusion (6:15-7:00)

> In summary, we fixed three critical bugs, built a complete quantum phase detection pipeline, and developed a novel phase-adapted VQE that achieves delta E below 0.12 for all phases. Our challenge-compliant noisy pipeline shows that ferromagnetic order is most noise-sensitive, with 29.7% degradation at p=0.05, while all phases remain correctly classified.

> The dual-signal approach — observable classifier plus variational basin — provides more robust phase detection than either signal alone.

> All results are reproducible from our Jupyter notebooks. Thank you.

*[Show: Final summary slide with repo link]*

---

## Images Needed

1. `structure_factor.png` — S(q) for three analytic states
2. `classifier_accuracy.png` — Before vs after bar chart
3. `vqe_comparison.png` — Standard vs phase-adapted VQE (log scale)
4. `noise_degradation.png` — Three-phase noise degradation plot
5. ANNNI phase diagram schematic (create manually in PowerPoint)
6. Phase-adapted HVA diagram (create manually in PowerPoint)
