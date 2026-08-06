import ObserverGeometry.RegularTree.Properties
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace RegularTree

noncomputable section

open Filter Topology

-- Regular Tree: Section 4, Limits.

-- THEOREM 12 (Small-decay limit).
theorem muRoot_tendsto_count_as_delta_tendsto_zero (k D R : ℕ) :
    Tendsto (fun δ => muRoot k δ D R) (𝓝 0)
      (𝓝 (∑ r ∈ Finset.Ico (R + 1) (D + 1), (k * (k - 1) ^ (r - 1) : ℝ))) := by
  unfold muRoot
  apply tendsto_finsetSum
  intro r _
  have hlin : Continuous (fun δ : ℝ => -δ * (r:ℝ)) := continuous_id.neg.mul continuous_const
  have hexp : Tendsto (fun δ : ℝ => Real.exp (-δ * r)) (𝓝 0) (𝓝 (Real.exp (-(0:ℝ) * r))) :=
    (Real.continuous_exp.comp hlin).tendsto 0
  have : Tendsto (fun δ : ℝ => (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r))
      (𝓝 0) (𝓝 ((k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-(0:ℝ) * r))) :=
    Tendsto.const_mul _ hexp
  simpa using this

-- THEOREM 13 (Infinite-tree convergence / phase transition).
def muRootTailTerm (k R : ℕ) (δ : ℝ) (i : ℕ) : ℝ :=
  (k:ℝ) * Real.exp (-δ) * (((k:ℝ) - 1) * Real.exp (-δ)) ^ (R + i)

theorem muRootTail_summable_iff (k R : ℕ) (δ : ℝ) (hk : 2 ≤ k) :
    Summable (muRootTailTerm k R δ) ↔ ((k:ℝ) - 1) * Real.exp (-δ) < 1 := by
  unfold muRootTailTerm
  set a : ℝ := ((k:ℝ) - 1) * Real.exp (-δ) with ha_def
  have hc_ne : (k:ℝ) * Real.exp (-δ) ≠ 0 := by positivity
  rw [summable_mul_left_iff hc_ne]
  rw [show (fun i => a ^ (R + i)) = (fun i => a ^ R * a ^ i) from
    funext (fun i => by rw [pow_add])]
  have hpowR_ne : a ^ R ≠ 0 := by
    apply pow_ne_zero
    have : (0:ℝ) < a := by
      rw [ha_def]
      have : (1:ℝ) ≤ (k:ℝ) - 1 := by
        have : (2:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk
        linarith
      positivity
    linarith
  rw [summable_mul_left_iff hpowR_ne]
  rw [summable_geometric_iff_norm_lt_one]
  have ha_nonneg : (0:ℝ) ≤ a := by
    rw [ha_def]
    have : (1:ℝ) ≤ (k:ℝ) - 1 := by
      have : (2:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk
      linarith
    positivity
  rw [Real.norm_eq_abs, abs_of_nonneg ha_nonneg]

theorem convergence_iff_delta_gt_log (k : ℕ) (δ : ℝ) (hk : 2 ≤ k) :
    ((k:ℝ) - 1) * Real.exp (-δ) < 1 ↔ δ > Real.log ((k:ℝ) - 1) := by
  have hkm1_pos : (0:ℝ) < (k:ℝ) - 1 := by
    have : (2:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk
    linarith
  have step1 : ((k:ℝ) - 1) * Real.exp (-δ) < 1 ↔ (k:ℝ) - 1 < Real.exp δ := by
    constructor
    · intro h
      have h2 := mul_lt_mul_of_pos_right h (Real.exp_pos δ)
      rwa [mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, mul_one, one_mul] at h2
    · intro h
      calc ((k:ℝ) - 1) * Real.exp (-δ)
          < Real.exp δ * Real.exp (-δ) := mul_lt_mul_of_pos_right h (Real.exp_pos (-δ))
        _ = 1 := by rw [← Real.exp_add]; simp
  rw [step1]
  have step2 : (k:ℝ) - 1 < Real.exp δ ↔ Real.log ((k:ℝ) - 1) < δ := by
    constructor
    · intro h
      have := Real.log_lt_log hkm1_pos h
      rwa [Real.log_exp] at this
    · intro h
      have := Real.exp_lt_exp.mpr h
      rwa [Real.exp_log hkm1_pos] at this
  rw [step2]

-- THEOREM 11 (Nearest-shell dominance).
theorem muRoot_nearest_shell_dominates (k D R : ℕ) (hRD : R < D) (hk : 1 ≤ k) :
    Tendsto (fun δ => muRoot k δ D R * Real.exp (δ * (R + 1)))
      atTop (𝓝 ((k * (k - 1) ^ R : ℕ) : ℝ)) := by
  unfold muRoot
  have heq : ∀ δ : ℝ, (∑ r ∈ Finset.Ico (R + 1) (D + 1),
      (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r)) * Real.exp (δ * (R + 1)) =
      ∑ r ∈ Finset.Ico (R + 1) (D + 1),
        (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * (r - (R + 1))) := by
    intro δ
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro r hr
    simp only [Finset.mem_Ico] at hr
    rw [mul_assoc, ← Real.exp_add]
    congr 2
    have : (R:ℝ) + 1 ≤ (r:ℝ) := by exact_mod_cast hr.1
    ring
  simp_rw [heq]
  have hreindex : ∀ δ : ℝ, ∑ r ∈ Finset.Ico (R + 1) (D + 1),
      (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * (r - (R + 1))) =
      ∑ i ∈ Finset.range (D + 1 - (R + 1)), (k * (k - 1) ^ (R + i) : ℝ) * Real.exp (-δ * i) := by
    intro δ
    apply Finset.sum_nbij' (fun r => r - (R + 1)) (fun i => R + 1 + i)
    · intro r hr
      simp only [Finset.mem_Ico] at hr
      simp only [Finset.mem_range]
      omega
    · intro i hi
      simp only [Finset.mem_range] at hi
      simp only [Finset.mem_Ico]
      omega
    · intro r hr
      simp only [Finset.mem_Ico] at hr
      omega
    · intro i hi
      simp only [Finset.mem_range] at hi
      omega
    · intro r hr
      simp only [Finset.mem_Ico] at hr
      have hpow_eq : r - 1 = R + (r - (R + 1)) := by omega
      have hexp_eq : (r:ℝ) - ((R:ℝ) + 1) = ((r - (R + 1) : ℕ) : ℝ) := by
        have hle : R + 1 ≤ r := hr.1
        push_cast [Nat.cast_sub hle]
        ring
      rw [hpow_eq, hexp_eq]
  simp_rw [hreindex]
  have hj0 : (0:ℕ) ∈ Finset.range (D + 1 - (R + 1)) := by
    simp only [Finset.mem_range]
    omega
  simp_rw [← Finset.sum_erase_add _ _ hj0]
  simp only [Nat.add_zero, Nat.cast_zero, mul_zero, Real.exp_zero, mul_one]
  have hterm : ∀ j ∈ (Finset.range (D + 1 - (R + 1))).erase 0,
      Tendsto (fun δ : ℝ => (k * (k - 1) ^ (R + j) : ℝ) * Real.exp (-δ * (j:ℝ)))
        atTop (𝓝 0) := by
    intro j hj
    simp only [Finset.mem_erase] at hj
    have hjpos : 0 < j := Nat.pos_of_ne_zero hj.1
    have hexp0 : Tendsto (fun δ : ℝ => Real.exp (-δ * (j:ℝ))) atTop (𝓝 0) := by
      apply Real.tendsto_exp_atBot.comp
      rw [tendsto_atBot]
      intro b
      filter_upwards [eventually_ge_atTop ((-b) / (j:ℝ))] with δ hδ
      have hjR : (0:ℝ) < (j:ℝ) := by exact_mod_cast hjpos
      have hkey : (-b / (j:ℝ)) * (j:ℝ) = -b := by field_simp
      have hge : δ * (j:ℝ) ≥ (-b / (j:ℝ)) * (j:ℝ) := mul_le_mul_of_nonneg_right hδ (le_of_lt hjR)
      rw [hkey] at hge
      linarith
    have hmul := Tendsto.const_mul ((k * (k - 1) ^ (R + j) : ℝ)) hexp0
    simpa using hmul
  have hlimrest : Tendsto (fun δ : ℝ =>
      ∑ j ∈ (Finset.range (D + 1 - (R + 1))).erase 0,
        (k * (k - 1) ^ (R + j) : ℝ) * Real.exp (-δ * j)) atTop (𝓝 0) := by
    simpa using tendsto_finsetSum _ hterm
  have hcombined := hlimrest.add_const ((k * (k - 1) ^ R : ℕ) : ℝ)
  push_cast [Nat.cast_sub hk] at hcombined ⊢
  simpa using hcombined

end

end RegularTree
