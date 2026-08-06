import ObserverGeometry.RegularTree.ExactDescriptor

namespace RegularTree

noncomputable section

-- Regular Tree: Section 3, Analytical Properties (Monotonicity).

-- Shared nonnegativity fact for a single muRoot summand, requiring
-- k ≥ 1 (needed since (k-1) could otherwise make the real-valued
-- factor sign ambiguous to automation).
theorem muRoot_term_nonneg (k : ℕ) (δ : ℝ) (hk : 1 ≤ k) (r : ℕ) (hr : 1 ≤ r) :
    0 ≤ (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r) := by
  rw [muRoot_term_eq k δ hk r hr]
  have hk1 : (0:ℝ) ≤ (k:ℝ) - 1 := by
    have : (1:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk
    linarith
  have ha_nonneg : (0:ℝ) ≤ ((k:ℝ) - 1) * Real.exp (-δ) := mul_nonneg hk1 (le_of_lt (Real.exp_pos _))
  exact mul_nonneg (mul_nonneg (Nat.cast_nonneg k) (le_of_lt (Real.exp_pos _)))
    (pow_nonneg ha_nonneg _)

-- THEOREM 7. Monotone in hidden depth R.
theorem muRoot_antitone_R (k D : ℕ) (δ : ℝ) (hk : 1 ≤ k) {R1 R2 : ℕ}
    (hR : R1 ≤ R2) (_hR2D : R2 ≤ D) :
    muRoot k δ D R2 ≤ muRoot k δ D R1 := by
  unfold muRoot
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro r hr
    simp only [Finset.mem_Ico] at hr ⊢
    omega
  · intro r hr _
    simp only [Finset.mem_Ico] at hr
    exact muRoot_term_nonneg k δ hk r (by omega)

-- THEOREM 8. Monotone in tree depth D.
theorem muRoot_mono_D (k R : ℕ) (δ : ℝ) (hk : 1 ≤ k) {D1 D2 : ℕ}
    (hD : D1 ≤ D2) (_hRD1 : R ≤ D1) :
    muRoot k δ D1 R ≤ muRoot k δ D2 R := by
  unfold muRoot
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro r hr
    simp only [Finset.mem_Ico] at hr ⊢
    omega
  · intro r hr _
    simp only [Finset.mem_Ico] at hr
    exact muRoot_term_nonneg k δ hk r (by omega)

-- THEOREM 9. Monotone in decay rate δ.
theorem muRoot_antitone_delta (k D R : ℕ) {δ1 δ2 : ℝ} (hδ : δ1 ≤ δ2) :
    muRoot k δ2 D R ≤ muRoot k δ1 D R := by
  unfold muRoot
  apply Finset.sum_le_sum
  intro r _
  have hnn : (0:ℝ) ≤ (k * (k - 1) ^ (r - 1) : ℝ) := by
    rcases Nat.eq_zero_or_pos k with hk0 | hkpos
    · simp [hk0]
    · have hk1 : 1 ≤ k := hkpos
      have hk1' : (0:ℝ) ≤ (k:ℝ) - 1 := by
        have : (1:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk1
        linarith
      positivity
  apply mul_le_mul_of_nonneg_left _ hnn
  apply Real.exp_le_exp.mpr
  have : (0:ℝ) ≤ (r:ℝ) := Nat.cast_nonneg r
  nlinarith

-- THEOREM 10. Monotone in branching factor k.
theorem muRoot_mono_k (D R : ℕ) (δ : ℝ) {k1 k2 : ℕ}
    (hk : k1 ≤ k2) (hk1 : 1 ≤ k1) :
    muRoot k1 δ D R ≤ muRoot k2 δ D R := by
  unfold muRoot
  apply Finset.sum_le_sum
  intro r _
  apply mul_le_mul_of_nonneg_right _ (le_of_lt (Real.exp_pos _))
  have hcast : (k1:ℝ) ≤ (k2:ℝ) := by exact_mod_cast hk
  have hcast1 : (1:ℝ) ≤ (k1:ℝ) := by exact_mod_cast hk1
  have hpow_nonneg : (0:ℝ) ≤ ((k1:ℝ) - 1) ^ (r - 1) := pow_nonneg (by linarith) _
  have hpow_mono : ((k1:ℝ) - 1) ^ (r - 1) ≤ ((k2:ℝ) - 1) ^ (r - 1) := by
    apply pow_le_pow_left₀ (by linarith) (by linarith)
  calc (k1:ℝ) * ((k1:ℝ) - 1) ^ (r - 1)
      ≤ (k2:ℝ) * ((k1:ℝ) - 1) ^ (r - 1) := by
        apply mul_le_mul_of_nonneg_right hcast hpow_nonneg
    _ ≤ (k2:ℝ) * ((k2:ℝ) - 1) ^ (r - 1) := by
        apply mul_le_mul_of_nonneg_left hpow_mono (by linarith)

end

end RegularTree
