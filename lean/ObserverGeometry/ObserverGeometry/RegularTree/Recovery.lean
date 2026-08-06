import ObserverGeometry.RegularTree.ExactDescriptor
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.RingTheory.Polynomial.Basic

namespace RegularTree

noncomputable section

-- Regular Tree: Section 7, Recovery (Theorems 19-20).

variable (R D : ℕ) (N N' : ℕ → ℝ)

def shellSum (δ : ℝ) : ℝ := ∑ r ∈ Finset.Ico (R + 1) (D + 1), N r * Real.exp (-δ * r)

-- THEOREMS 19 & 20 (Recovery / Injectivity -- one proof, two readings).
theorem shellCoeffs_eq_of_shellSum_eq
    (h : ∀ δ : ℝ, shellSum R D N δ = shellSum R D N' δ) :
    ∀ r ∈ Finset.Ico (R + 1) (D + 1), N r = N' r := by
  set P : Polynomial ℝ :=
    ∑ r ∈ Finset.Ico (R + 1) (D + 1), Polynomial.C (N r - N' r) * Polynomial.X ^ r
    with hP_def
  have heval : ∀ x : ℝ, 0 < x → P.eval x = 0 := by
    intro x hx
    have hkey := h (-Real.log x)
    unfold shellSum at hkey
    simp only [neg_neg] at hkey
    have heq : P.eval x = ∑ r ∈ Finset.Ico (R + 1) (D + 1), (N r - N' r) * x ^ r := by
      rw [hP_def]
      simp [Polynomial.eval_finsetSum]
    rw [heq]
    have hrw : ∀ r ∈ Finset.Ico (R + 1) (D + 1), (N r - N' r) * x ^ r
        = N r * x ^ r - N' r * x ^ r := by intro r _; ring
    rw [Finset.sum_congr rfl hrw, Finset.sum_sub_distrib]
    have hxeq : ∀ r : ℕ, x ^ r = Real.exp (Real.log x * r) := by
      intro r
      conv_lhs => rw [← Real.exp_log hx]
      rw [← Real.exp_nat_mul]
      ring_nf
    simp_rw [hxeq]
    linarith [hkey]
  have hinf : {x : ℝ | P.IsRoot x}.Infinite := by
    apply Set.Infinite.mono (s := Set.Ioi (0:ℝ))
    · intro x hx
      exact heval x hx
    · exact Set.Ioi_infinite 0
  have hP0 : P = 0 := Polynomial.eq_zero_of_infinite_isRoot P hinf
  intro r hr
  have hcoeff : P.coeff r = N r - N' r := by
    rw [hP_def, Polynomial.finsetSum_coeff]
    rw [Finset.sum_eq_single r]
    · rw [Polynomial.coeff_C_mul, Polynomial.coeff_X_pow_self]
      ring
    · intro r' hr' hne
      rw [Polynomial.coeff_C_mul, Polynomial.coeff_X_pow, if_neg (Ne.symm hne)]
      ring
    · intro hcontra
      exact absurd hr hcontra
  rw [hP0] at hcoeff
  simp only [Polynomial.coeff_zero] at hcoeff
  linarith [hcoeff]

-- NOTE: Theorem 19 (Recovery) and Theorem 20 (Injectivity) are the
-- SAME theorem read two ways -- "the shell counts are recoverable
-- from the descriptor curve" and "different shell-count profiles
-- give different descriptor curves" are literally the same
-- statement, contrapositive-flipped. There is deliberately only one
-- proof (shellCoeffs_eq_of_shellSum_eq above), not two.

end

end RegularTree
