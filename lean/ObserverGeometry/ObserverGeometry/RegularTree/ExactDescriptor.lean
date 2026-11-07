import ObserverGeometry.RegularTree.Geometry
import Mathlib.Analysis.SpecialFunctions.Exp

namespace RegularTree

noncomputable section

-- Closed-form remaining geometry at the root of the k-regular tree (Theorem 2(a)). `muRoot` is
-- defined by the shell-count formula; `FullBridge` identifies it with the general `mu` on the
-- tree metric space and with the framework descriptor `Phi`.

theorem exp_nat_mul_helper (δ : ℝ) (r : ℕ) : Real.exp (-δ * r) = Real.exp (-δ) ^ r := by
  rw [← Real.exp_nat_mul]
  ring_nf

def muRoot (k : ℕ) (δ : ℝ) (D R : ℕ) : ℝ :=
  ∑ r ∈ Finset.Ico (R + 1) (D + 1), (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r)

-- Shared per-term rewriting: k(k-1)^(r-1) exp(-δr) = k exp(-δ) a^(r-1).
theorem muRoot_term_eq (k : ℕ) (δ : ℝ) (_hk : 1 ≤ k) (r : ℕ) (hr : 1 ≤ r) :
    (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r) =
      (k : ℝ) * Real.exp (-δ) * (((k : ℝ) - 1) * Real.exp (-δ)) ^ (r - 1) := by
  have hrsub : r = (r - 1) + 1 := by omega
  rw [mul_pow, exp_nat_mul_helper]
  conv_lhs => rw [hrsub]
  rw [pow_succ]
  push_cast [Nat.cast_sub _hk]
  ring

-- THEOREM 5 (Closed form). a := (k-1) e^{-δ}. Matches the boxed
-- formula mu = k e^{-δ} (a^R - a^D) / (1 - a) exactly.
theorem muRoot_closed_form (k D R : ℕ) (δ : ℝ) (hk : 1 ≤ k) (hR : R ≤ D)
    (ha : ((k : ℝ) - 1) * Real.exp (-δ) ≠ 1) :
    muRoot k δ D R =
      (k : ℝ) * Real.exp (-δ) *
        ((((k : ℝ) - 1) * Real.exp (-δ)) ^ R - (((k : ℝ) - 1) * Real.exp (-δ)) ^ D) /
          (1 - ((k : ℝ) - 1) * Real.exp (-δ)) := by
  unfold muRoot
  set a : ℝ := ((k : ℝ) - 1) * Real.exp (-δ) with ha_def
  have hterm : ∀ r ∈ Finset.Ico (R + 1) (D + 1),
      (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r) = (k : ℝ) * Real.exp (-δ) * a ^ (r - 1) := by
    intro r hr
    simp only [Finset.mem_Ico] at hr
    rw [ha_def]
    exact muRoot_term_eq k δ hk r (by omega)
  rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum]
  have hreindex : ∑ r ∈ Finset.Ico (R + 1) (D + 1), a ^ (r - 1) =
      ∑ i ∈ Finset.Ico R D, a ^ i := by
    apply Finset.sum_nbij' (fun r => r - 1) (fun i => i + 1)
    · intro r hr; simp only [Finset.mem_Ico] at hr ⊢; omega
    · intro i hi; simp only [Finset.mem_Ico] at hi ⊢; omega
    · intro r hr; simp only [Finset.mem_Ico] at hr; omega
    · intro i hi; simp only [Finset.mem_Ico] at hi; omega
    · intro r hr; rfl
  rw [hreindex, geom_sum_Ico ha hR]
  field_simp
  ring

-- THEOREM 6 (Critical case).
theorem muRoot_critical (k D R : ℕ) (δ : ℝ) (hk : 1 ≤ k) (hR : R ≤ D)
    (ha : ((k : ℝ) - 1) * Real.exp (-δ) = 1) :
    muRoot k δ D R = (k : ℝ) * Real.exp (-δ) * (D - R) := by
  unfold muRoot
  have hterm : ∀ r ∈ Finset.Ico (R + 1) (D + 1),
      (k * (k - 1) ^ (r - 1) : ℝ) * Real.exp (-δ * r) = (k : ℝ) * Real.exp (-δ) := by
    intro r hr
    simp only [Finset.mem_Ico] at hr
    rw [muRoot_term_eq k δ hk r (by omega), ha, one_pow, mul_one]
  rw [Finset.sum_congr rfl hterm, Finset.sum_const, Nat.card_Ico, nsmul_eq_mul]
  push_cast [hR]
  ring

end

end RegularTree
