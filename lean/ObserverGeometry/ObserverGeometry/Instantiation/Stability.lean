import ObserverGeometry.Instantiation.Mu

namespace Instantiation

noncomputable section

-- Stability theorems for mu.

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

-- STABILITY I. H' = H ∪ A ==> mu(H') - mu(H) = mu(A \ H).
omit [Fintype V] in
theorem mu_stability_I (δ : ℝ) (v : V) (H A : Finset V) :
    mu f δ (H ∪ A) v - mu f δ H v = mu f δ (A \ H) v := by
  have hdisj : Disjoint H (A \ H) := by
    rw [Finset.disjoint_left]
    intro x hx hx2
    simp only [Finset.mem_sdiff] at hx2
    exact hx2.2 hx
  have heq : H ∪ A = H ∪ (A \ H) := by
    ext x; simp only [Finset.mem_union, Finset.mem_sdiff]; tauto
  rw [heq, mu_additive f δ v H (A \ H) hdisj]
  ring

omit [Fintype V] in
theorem mu_stability_I_bound (δ : ℝ) (hδ : 0 ≤ δ) (v : V) (H A : Finset V) :
    |mu f δ (H ∪ A) v - mu f δ H v| ≤ (A.card : ℝ) := by
  rw [mu_stability_I f δ v H A]
  rw [abs_of_nonneg (mu_nonneg f δ (A \ H) v)]
  unfold mu
  calc ∑ u ∈ A \ H, Real.exp (-δ * dist (f v) (f u))
      ≤ ∑ _u ∈ A \ H, (1:ℝ) := by
        apply Finset.sum_le_sum
        intro u _
        rw [Real.exp_le_one_iff]
        have := dist_nonneg (x := f v) (y := f u)
        nlinarith
    _ = (A \ H).card := by rw [Finset.sum_const, nsmul_eq_mul, mul_one]
    _ ≤ (A.card : ℝ) := by
        exact_mod_cast Finset.card_le_card (Finset.sdiff_subset)

-- STABILITY II. If two hidden regions differ by at most k elements
-- (counting both directions of the symmetric difference), mu differs
-- by at most k.
omit [Fintype V] in
theorem mu_stability_II (δ : ℝ) (hδ : 0 ≤ δ) (v : V) (H H' : Finset V) (k : ℕ)
    (hk : (H \ H').card + (H' \ H).card ≤ k) :
    |mu f δ H v - mu f δ H' v| ≤ (k : ℝ) := by
  have hHeq : H = (H ∩ H') ∪ (H \ H') := by
    ext x; simp only [Finset.mem_union, Finset.mem_inter, Finset.mem_sdiff]; tauto
  have hH'eq : H' = (H ∩ H') ∪ (H' \ H) := by
    ext x; simp only [Finset.mem_union, Finset.mem_inter, Finset.mem_sdiff]; tauto
  have hdisjH : Disjoint (H ∩ H') (H \ H') := by
    rw [Finset.disjoint_left]; intro x hx hx2
    simp only [Finset.mem_inter] at hx
    simp only [Finset.mem_sdiff] at hx2
    exact hx2.2 hx.2
  have hdisjH' : Disjoint (H ∩ H') (H' \ H) := by
    rw [Finset.disjoint_left]; intro x hx hx2
    simp only [Finset.mem_inter] at hx
    simp only [Finset.mem_sdiff] at hx2
    exact hx2.2 hx.1
  have heqH : mu f δ H v = mu f δ (H ∩ H') v + mu f δ (H \ H') v := by
    conv_lhs => rw [hHeq]
    exact mu_additive f δ v (H ∩ H') (H \ H') hdisjH
  have heqH' : mu f δ H' v = mu f δ (H ∩ H') v + mu f δ (H' \ H) v := by
    conv_lhs => rw [hH'eq]
    exact mu_additive f δ v (H ∩ H') (H' \ H) hdisjH'
  have hbound1 : mu f δ (H \ H') v ≤ ((H \ H').card : ℝ) := by
    unfold mu
    calc ∑ u ∈ H \ H', Real.exp (-δ * dist (f v) (f u))
        ≤ ∑ _u ∈ H \ H', (1:ℝ) := by
          apply Finset.sum_le_sum; intro u _
          rw [Real.exp_le_one_iff]; nlinarith [dist_nonneg (x := f v) (y := f u)]
      _ = (H \ H').card := by rw [Finset.sum_const, nsmul_eq_mul, mul_one]
  have hbound2 : mu f δ (H' \ H) v ≤ ((H' \ H).card : ℝ) := by
    unfold mu
    calc ∑ u ∈ H' \ H, Real.exp (-δ * dist (f v) (f u))
        ≤ ∑ _u ∈ H' \ H, (1:ℝ) := by
          apply Finset.sum_le_sum; intro u _
          rw [Real.exp_le_one_iff]; nlinarith [dist_nonneg (x := f v) (y := f u)]
      _ = (H' \ H).card := by rw [Finset.sum_const, nsmul_eq_mul, mul_one]
  have hkR : ((H \ H').card : ℝ) + ((H' \ H).card : ℝ) ≤ (k : ℝ) := by exact_mod_cast hk
  rw [abs_le]
  constructor
  · rw [heqH, heqH']
    nlinarith [hbound1, hbound2, mu_nonneg f δ (H \ H') v, mu_nonneg f δ (H' \ H) v, hkR]
  · rw [heqH, heqH']
    nlinarith [hbound1, hbound2, mu_nonneg f δ (H \ H') v, mu_nonneg f δ (H' \ H) v, hkR]

-- Helper lemma (Lipschitz bound on the exponential kernel; adapted
-- from the pattern already proven correct earlier in this project).
theorem exp_dist_lipschitz (δ a b bound : ℝ) (hδ : 0 ≤ δ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hbound : |a - b| ≤ bound) :
    |Real.exp (-δ * a) - Real.exp (-δ * b)| ≤ δ * bound := by
  have hab_le : |a - b| ≤ bound := hbound
  rcases le_total a b with hle | hle
  · -- a ≤ b: exp(-δa) ≥ exp(-δb), so exp(-δa)-exp(-δb) ≥ 0.
    have h1 : Real.exp (-δ * b) ≤ Real.exp (-δ * a) := by
      apply Real.exp_le_exp.mpr; nlinarith
    have hkey : Real.exp (-δ * a) - Real.exp (-δ * b)
        = Real.exp (-δ * a) * (1 - Real.exp (-δ * (b - a))) := by
      rw [mul_sub, mul_one, ← Real.exp_add]; ring_nf
    have hb1 : Real.exp (-δ * a) ≤ 1 := by rw [Real.exp_le_one_iff]; nlinarith
    have hb2 : 1 - Real.exp (-δ * (b - a)) ≤ δ * (b - a) := by
      have := Real.add_one_le_exp (-δ * (b - a)); nlinarith
    have hb3 : 0 ≤ 1 - Real.exp (-δ * (b - a)) := by
      have : Real.exp (-δ * (b - a)) ≤ 1 := by rw [Real.exp_le_one_iff]; nlinarith
      linarith
    have hle2 : Real.exp (-δ * a) - Real.exp (-δ * b) ≤ δ * (b - a) := by
      rw [hkey]; nlinarith [Real.exp_pos (-δ * a)]
    rw [abs_of_nonneg (by linarith : (0:ℝ) ≤ Real.exp (-δ * a) - Real.exp (-δ * b))]
    have hdiff : b - a ≤ bound := by rw [abs_le] at hab_le; linarith [hab_le.1]
    nlinarith [mul_le_mul_of_nonneg_left hdiff hδ]
  · -- b ≤ a: exp(-δb) ≥ exp(-δa), so exp(-δa)-exp(-δb) ≤ 0.
    have h1 : Real.exp (-δ * a) ≤ Real.exp (-δ * b) := by
      apply Real.exp_le_exp.mpr; nlinarith
    have hkey : Real.exp (-δ * b) - Real.exp (-δ * a)
        = Real.exp (-δ * b) * (1 - Real.exp (-δ * (a - b))) := by
      rw [mul_sub, mul_one, ← Real.exp_add]; ring_nf
    have hb1 : Real.exp (-δ * b) ≤ 1 := by rw [Real.exp_le_one_iff]; nlinarith
    have hb2 : 1 - Real.exp (-δ * (a - b)) ≤ δ * (a - b) := by
      have := Real.add_one_le_exp (-δ * (a - b)); nlinarith
    have hb3 : 0 ≤ 1 - Real.exp (-δ * (a - b)) := by
      have : Real.exp (-δ * (a - b)) ≤ 1 := by rw [Real.exp_le_one_iff]; nlinarith
      linarith
    have hle2 : Real.exp (-δ * b) - Real.exp (-δ * a) ≤ δ * (a - b) := by
      rw [hkey]; nlinarith [Real.exp_pos (-δ * b)]
    rw [abs_of_nonpos (by linarith : Real.exp (-δ * a) - Real.exp (-δ * b) ≤ 0)]
    have hdiff : a - b ≤ bound := by rw [abs_le] at hab_le; linarith [hab_le.2]
    nlinarith [mul_le_mul_of_nonneg_left hdiff hδ]

-- STABILITY III. If every hidden-node distance changes by at most
-- epsilon between two embeddings f, g, mu changes by at most
-- delta * |H| * epsilon.
omit [Fintype V] [DecidableEq V] in
theorem mu_stability_III (δ ε : ℝ) (hδ : 0 ≤ δ) (v : V) (H : Finset V)
    (g : V → X) (hclose : ∀ u ∈ H, |dist (f v) (f u) - dist (g v) (g u)| ≤ ε) :
    |mu f δ H v - (∑ u ∈ H, Real.exp (-δ * dist (g v) (g u)))| ≤ δ * H.card * ε := by
  unfold mu
  calc |∑ u ∈ H, Real.exp (-δ * dist (f v) (f u)) -
        ∑ u ∈ H, Real.exp (-δ * dist (g v) (g u))|
      = |∑ u ∈ H, (Real.exp (-δ * dist (f v) (f u)) - Real.exp (-δ * dist (g v) (g u)))| := by
        rw [Finset.sum_sub_distrib]
    _ ≤ ∑ u ∈ H, |Real.exp (-δ * dist (f v) (f u)) - Real.exp (-δ * dist (g v) (g u))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _u ∈ H, δ * ε := by
        apply Finset.sum_le_sum
        intro u hu
        exact exp_dist_lipschitz δ (dist (f v) (f u)) (dist (g v) (g u)) ε hδ
          dist_nonneg dist_nonneg (hclose u hu)
    _ = H.card * (δ * ε) := by rw [Finset.sum_const, nsmul_eq_mul]
    _ = δ * H.card * ε := by ring

-- STABILITY V. Observer moves from v to v': |mu_H(v) - mu_H(v')|
-- ≤ |H| * delta * d(v, v').
omit [Fintype V] [DecidableEq V] in
theorem mu_stability_V (δ : ℝ) (hδ : 0 ≤ δ) (H : Finset V) (v v' : V) :
    |mu f δ H v - mu f δ H v'| ≤ H.card * δ * dist (f v) (f v') := by
  unfold mu
  calc |∑ u ∈ H, Real.exp (-δ * dist (f v) (f u)) -
        ∑ u ∈ H, Real.exp (-δ * dist (f v') (f u))|
      = |∑ u ∈ H, (Real.exp (-δ * dist (f v) (f u)) - Real.exp (-δ * dist (f v') (f u)))| := by
        rw [Finset.sum_sub_distrib]
    _ ≤ ∑ u ∈ H, |Real.exp (-δ * dist (f v) (f u)) - Real.exp (-δ * dist (f v') (f u))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _u ∈ H, δ * dist (f v) (f v') := by
        apply Finset.sum_le_sum
        intro u _
        have htri : |dist (f v) (f u) - dist (f v') (f u)| ≤ dist (f v) (f v') := by
          have h1 : dist (f v) (f u) ≤ dist (f v) (f v') + dist (f v') (f u) := dist_triangle _ _ _
          have h2 : dist (f v') (f u) ≤ dist (f v') (f v) + dist (f v) (f u) := dist_triangle _ _ _
          rw [abs_le]
          constructor
          · linarith [dist_comm (f v') (f v)]
          · linarith
        exact exp_dist_lipschitz δ (dist (f v) (f u)) (dist (f v') (f u)) (dist (f v) (f v'))
          hδ dist_nonneg dist_nonneg htri
    _ = H.card * (δ * dist (f v) (f v')) := by rw [Finset.sum_const, nsmul_eq_mul]
    _ = H.card * δ * dist (f v) (f v') := by ring

-- STABILITY VI (Embedding Equivalence Stability). If two embeddings
-- induce IDENTICAL distances from v to every hidden node, mu is
-- exactly equal -- not merely close. This is strictly stronger than
-- coordinate invariance: it doesn't require a global isometry of the
-- ambient space, only that the specific distances used by mu happen
-- to coincide. Direct corollary of Stability III with epsilon = 0.
omit [Fintype V] [DecidableEq V] in
theorem mu_stability_VI (δ : ℝ) (hδ : 0 ≤ δ) (v : V) (H : Finset V)
    (g : V → X) (heq : ∀ u ∈ H, dist (f v) (f u) = dist (g v) (g u)) :
    mu f δ H v = ∑ u ∈ H, Real.exp (-δ * dist (g v) (g u)) := by
  have h := mu_stability_III f δ 0 hδ v H g (fun u hu => by rw [heq u hu]; simp)
  simp only [mul_zero, abs_le] at h
  linarith [h.1, h.2]

-- STABILITY VII. Only one hidden node changes: |Δmu| = exp(-δ d).
omit [Fintype V] in
theorem mu_stability_VII (δ : ℝ) (v u : V) (H : Finset V) (hu : u ∉ H) :
    mu f δ (insert u H) v - mu f δ H v = Real.exp (-δ * dist (f v) (f u)) := by
  unfold mu
  rw [Finset.sum_insert hu]
  ring

end

end Instantiation
