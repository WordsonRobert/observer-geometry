import ObserverGeometry.RegularTree.ExactDescriptor
import Mathlib.Analysis.SpecificLimits.Basic

namespace RegularTree

noncomputable section

open Filter Topology

-- Regular Tree: Section 8, Characterization (Theorems 21-24).

variable (k D R : ℕ) (δ : ℝ)

theorem muRoot_recurrence (hRD : R < D) :
    muRoot k δ D R = muRoot k δ D (R + 1) + (k * (k - 1) ^ R : ℝ) * Real.exp (-δ * (R + 1)) := by
  unfold muRoot
  rw [show Finset.Ico (R + 1) (D + 1) = insert (R + 1) (Finset.Ico (R + 2) (D + 1)) by
    ext x; simp only [Finset.mem_insert, Finset.mem_Ico]; omega]
  rw [Finset.sum_insert (by simp only [Finset.mem_Ico]; omega)]
  have hshift : Finset.Ico (R + 2) (D + 1) = Finset.Ico ((R + 1) + 1) (D + 1) := by
    congr 1
  rw [hshift]
  have hpow : (R + 1) - 1 = R := by omega
  rw [hpow]
  have hexp_eq : Real.exp (-δ * ((R + 1 : ℕ) : ℝ)) = Real.exp (-δ * ((R:ℝ) + 1)) := by
    congr 1
    push_cast
    ring
  rw [hexp_eq]
  ring

theorem muRoot_boundary : muRoot k δ D D = 0 := by
  unfold muRoot
  simp

theorem muRoot_uniqueness (f : ℕ → ℝ)
    (hbase : f D = 0)
    (hrec : ∀ R < D, f R = f (R + 1) + (k * (k - 1) ^ R : ℝ) * Real.exp (-δ * (R + 1))) :
    ∀ R ≤ D, f R = muRoot k δ D R := by
  have key : ∀ n : ℕ, ∀ R', R' + n = D → f R' = muRoot k δ D R' := by
    intro n
    induction n with
    | zero =>
      intro R' hRD
      have hRD' : R' = D := by omega
      subst hRD'
      rw [hbase, muRoot_boundary]
    | succ n ih =>
      intro R' hRD
      have hRlt : R' < D := by omega
      have hR1 : (R' + 1) + n = D := by omega
      rw [hrec R' hRlt, ih (R' + 1) hR1, ← muRoot_recurrence k D R' δ hRlt]
  intro R hR
  exact key (D - R) R (by omega)

-- THEOREM 24 (Asymptotic Growth). In the DIVERGENT regime
-- a := (k-1)e^{-δ} > 1, muRoot(D,R) grows exactly like a^D: the
-- ratio tends to the CONSTANT k e^{-δ}/(a-1), independent of R
-- (the fixed cutoff becomes negligible as D → ∞).
theorem muRoot_asymptotic_growth (hk : 1 ≤ k)
    (ha : 1 < ((k : ℝ) - 1) * Real.exp (-δ)) :
    Tendsto (fun D : ℕ => muRoot k δ D R / (((k : ℝ) - 1) * Real.exp (-δ)) ^ D)
      atTop (𝓝 ((k : ℝ) * Real.exp (-δ) / (((k : ℝ) - 1) * Real.exp (-δ) - 1))) := by
  set a : ℝ := ((k : ℝ) - 1) * Real.exp (-δ) with ha_def
  have ha_pos : 0 < a := by linarith
  have heq : ∀ D : ℕ, R ≤ D → muRoot k δ D R / a ^ D =
      (k : ℝ) * Real.exp (-δ) * (a ^ R - a ^ D) / (1 - a) / a ^ D := by
    intro D hD
    rw [muRoot_closed_form k D R δ hk hD (by linarith)]
  have hsimplify : ∀ D : ℕ, R ≤ D →
      (k : ℝ) * Real.exp (-δ) * (a ^ R - a ^ D) / (1 - a) / a ^ D =
      (k : ℝ) * Real.exp (-δ) * (a ^ R / a ^ D - 1) / (1 - a) := by
    intro D hD
    have hane : a ^ D ≠ 0 := pow_ne_zero _ (ne_of_gt ha_pos)
    field_simp
  have htendsto1 : Tendsto (fun D : ℕ => a ^ R / a ^ D) atTop (𝓝 0) := by
    have hlt1 : Tendsto (fun D : ℕ => (1 / a) ^ D) atTop (𝓝 0) := by
      apply tendsto_pow_atTop_nhds_zero_of_lt_one
      · positivity
      · rw [div_lt_one ha_pos]; linarith
    have heq2 : ∀ D : ℕ, a ^ R / a ^ D = a ^ R * (1 / a) ^ D := by
      intro D
      rw [div_pow, one_pow]
      ring
    simp_rw [heq2]
    have := Tendsto.const_mul (a ^ R) hlt1
    simpa using this
  have htendsto2 : Tendsto (fun D : ℕ =>
      (k : ℝ) * Real.exp (-δ) * (a ^ R / a ^ D - 1) / (1 - a)) atTop
      (𝓝 ((k : ℝ) * Real.exp (-δ) * (0 - 1) / (1 - a))) := by
    apply Tendsto.div_const
    apply Tendsto.const_mul
    exact Tendsto.sub_const htendsto1 1
  have hfinal_eq : (k : ℝ) * Real.exp (-δ) * (0 - 1) / (1 - a) =
      (k : ℝ) * Real.exp (-δ) / (a - 1) := by
    rw [zero_sub]
    rw [div_eq_div_iff (by linarith) (by linarith)]
    ring
  rw [← hfinal_eq]
  apply Tendsto.congr' _ htendsto2
  filter_upwards [eventually_ge_atTop R] with D hD
  rw [heq D hD, hsimplify D hD]

end

end RegularTree
