import ObserverGeometry.Instantiation.Stability

/-!
# Stability of the normalized remaining geometry

The reported descriptor is the normalized functional `μ(v, H) / μ(v, V)`.
This module proves dimension-free stability bounds for it.

* `ratio_gap`: if every weight changes by a factor of at most `l`, a normalized
  weight sum moves by at most `(l - 1) / (l + 1)`.
* `muN_stability_embedding`: if every distance from the observer changes by at most `η`
  between two embeddings, the normalized functional moves by at most
  `(exp (δ η) - 1) / (exp (δ η) + 1) = tanh (δ η / 2)`.
* `muN_stability_observer`: the same bound with `η = d(v, v')` when the observer moves
  and the hidden region is held fixed.

In contrast with `mu_stability_III` and `mu_stability_V`, the constants do not depend on
the size of the hidden region.
-/

namespace Instantiation

noncomputable section

open Finset

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

/-- Normalized remaining geometry `μ(v, H) / μ(v, V)`. -/
def muN (δ : ℝ) (H : Finset V) (v : V) : ℝ :=
  mu f δ H v / mu f δ Finset.univ v

/-- One-sided algebraic core: moving the numerator mass up by at most a factor `l` and the
complementary mass down by at most a factor `l` raises the fraction by at most
`(l - 1) / (l + 1)`. -/
theorem ratio_gap_upper (a b a' b' l : ℝ) (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a')
    (hb' : 0 < b') (hl : 1 ≤ l) (h1 : a' ≤ l * a) (h2 : b ≤ l * b') :
    a' / (a' + b') - a / (a + b) ≤ (l - 1) / (l + 1) := by
  have hl0 : 0 < l := by linarith
  -- step 1: a'/(a'+b') ≤ l^2 a / (l^2 a + b)
  have step1 : a' / (a' + b') ≤ (l ^ 2 * a) / (l ^ 2 * a + b) := by
    rw [div_le_div_iff₀ (by linarith) (by positivity)]
    have : a' * b ≤ l ^ 2 * a * b' := by
      calc a' * b ≤ (l * a) * b := by exact mul_le_mul_of_nonneg_right h1 hb.le
        _ ≤ (l * a) * (l * b') := by
            exact mul_le_mul_of_nonneg_left h2 (by positivity)
        _ = l ^ 2 * a * b' := by ring
    nlinarith
  -- step 2: l^2 a/(l^2 a + b) - a/(a+b) ≤ (l-1)/(l+1), equivalent to (l a - b)^2 ≥ 0
  have step2 : (l ^ 2 * a) / (l ^ 2 * a + b) - a / (a + b) ≤ (l - 1) / (l + 1) := by
    rw [div_sub_div _ _ (by positivity) (by positivity),
      div_le_div_iff₀ (by positivity) (by positivity)]
    have hsq : 0 ≤ (l * a - b) ^ 2 := sq_nonneg _
    have hlm : 0 ≤ l - 1 := by linarith
    nlinarith [mul_nonneg hlm hsq, mul_pos ha hb, mul_nonneg hlm (mul_pos ha hb).le]
  linarith

/-- Two-sided version of `ratio_gap_upper`. -/
theorem ratio_gap (a b a' b' l : ℝ) (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a')
    (hb' : 0 < b') (hl : 1 ≤ l)
    (h1 : a' ≤ l * a) (h1' : a ≤ l * a') (h2 : b' ≤ l * b) (h2' : b ≤ l * b') :
    |a' / (a' + b') - a / (a + b)| ≤ (l - 1) / (l + 1) := by
  rw [abs_le]
  constructor
  · have := ratio_gap_upper a' b' a b l ha' hb' ha hb hl h1' h2
    linarith
  · exact ratio_gap_upper a b a' b' l ha hb ha' hb' hl h1 h2'

/-- Termwise multiplicative comparison of the exponential kernel. -/
theorem exp_kernel_ratio (δ η x y : ℝ) (hδ : 0 ≤ δ) (h : |x - y| ≤ η) :
    Real.exp (-δ * y) ≤ Real.exp (δ * η) * Real.exp (-δ * x) := by
  rw [← Real.exp_add]
  apply Real.exp_le_exp.mpr
  have : x - y ≤ η := (abs_le.mp h).2
  nlinarith

omit [Fintype V] [DecidableEq V] in
/-- Sum version: weights `w2 ≤ l * w1` termwise give `Σ w2 ≤ l * Σ w1`. -/
theorem sum_le_mul_sum (S : Finset V) (w1 w2 : V → ℝ) (l : ℝ)
    (h : ∀ u ∈ S, w2 u ≤ l * w1 u) : ∑ u ∈ S, w2 u ≤ l * ∑ u ∈ S, w1 u := by
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum h

omit [Fintype V] [DecidableEq V] in
theorem mu_pos_of_nonempty (δ : ℝ) (H : Finset V) (v : V) (hH : H.Nonempty) :
    0 < mu f δ H v := by
  unfold mu
  exact Finset.sum_pos (fun _ _ => Real.exp_pos _) hH

theorem mu_univ_split (δ : ℝ) (H : Finset V) (v : V) :
    mu f δ Finset.univ v = mu f δ H v + mu f δ (Finset.univ \ H) v := by
  have hdisj : Disjoint H (Finset.univ \ H) := Finset.disjoint_sdiff
  rw [← mu_additive f δ v H (Finset.univ \ H) hdisj, Finset.union_sdiff_of_subset
    (Finset.subset_univ H)]

/-- Abstract normalized stability for two weightings of the same finite set. -/
theorem normalized_ratio_stability (H : Finset V) (w1 w2 : V → ℝ) (l : ℝ) (hl : 1 ≤ l)
    (hpos1 : ∀ u, 0 < w1 u) (hpos2 : ∀ u, 0 < w2 u)
    (h12 : ∀ u, w2 u ≤ l * w1 u) (h21 : ∀ u, w1 u ≤ l * w2 u)
    (hH : H.Nonempty) (hHc : (Finset.univ \ H).Nonempty) :
    |(∑ u ∈ H, w2 u) / ((∑ u ∈ H, w2 u) + ∑ u ∈ Finset.univ \ H, w2 u)
      - (∑ u ∈ H, w1 u) / ((∑ u ∈ H, w1 u) + ∑ u ∈ Finset.univ \ H, w1 u)|
      ≤ (l - 1) / (l + 1) := by
  apply ratio_gap
  · exact Finset.sum_pos (fun u _ => hpos1 u) hH
  · exact Finset.sum_pos (fun u _ => hpos1 u) hHc
  · exact Finset.sum_pos (fun u _ => hpos2 u) hH
  · exact Finset.sum_pos (fun u _ => hpos2 u) hHc
  · exact hl
  · exact sum_le_mul_sum H w1 w2 l (fun u _ => h12 u)
  · exact sum_le_mul_sum H w2 w1 l (fun u _ => h21 u)
  · exact sum_le_mul_sum _ w1 w2 l (fun u _ => h12 u)
  · exact sum_le_mul_sum _ w2 w1 l (fun u _ => h21 u)

/-- **Normalized embedding stability.** If every distance from the observer changes by at
most `η` between embeddings `f` and `g`, the normalized remaining geometry of any hidden
region changes by at most `(exp (δ η) - 1) / (exp (δ η) + 1)`, independently of `|H|`. -/
theorem muN_stability_embedding (δ η : ℝ) (hδ : 0 ≤ δ) (hη : 0 ≤ η) (v : V) (H : Finset V)
    (g : V → X) (hclose : ∀ u, |dist (f v) (f u) - dist (g v) (g u)| ≤ η)
    (hH : H.Nonempty) (hHc : (Finset.univ \ H).Nonempty) :
    |muN g δ H v - muN f δ H v| ≤ (Real.exp (δ * η) - 1) / (Real.exp (δ * η) + 1) := by
  have hl : 1 ≤ Real.exp (δ * η) := Real.one_le_exp (mul_nonneg hδ hη)
  unfold muN
  rw [mu_univ_split g δ H v, mu_univ_split f δ H v]
  unfold mu
  apply normalized_ratio_stability H (fun u => Real.exp (-δ * dist (f v) (f u)))
    (fun u => Real.exp (-δ * dist (g v) (g u))) _ hl
  · intro u; exact Real.exp_pos _
  · intro u; exact Real.exp_pos _
  · intro u; exact exp_kernel_ratio δ η _ _ hδ (hclose u)
  · intro u
    have h := hclose u
    rw [abs_sub_comm] at h
    exact exp_kernel_ratio δ η _ _ hδ h
  · exact hH
  · exact hHc

/-- **Normalized observer stability.** For a fixed hidden region, moving the observer from
`v` to `v'` changes the normalized remaining geometry by at most
`(exp (δ d) - 1) / (exp (δ d) + 1)` with `d = dist (f v) (f v')`. -/
theorem muN_stability_observer (δ : ℝ) (hδ : 0 ≤ δ) (H : Finset V) (v v' : V)
    (hH : H.Nonempty) (hHc : (Finset.univ \ H).Nonempty) :
    |muN f δ H v' - muN f δ H v|
      ≤ (Real.exp (δ * dist (f v) (f v')) - 1) / (Real.exp (δ * dist (f v) (f v')) + 1) := by
  set d := dist (f v) (f v')
  have hd : 0 ≤ d := dist_nonneg
  have hl : 1 ≤ Real.exp (δ * d) := Real.one_le_exp (mul_nonneg hδ hd)
  have htri : ∀ u, |dist (f v) (f u) - dist (f v') (f u)| ≤ d := by
    intro u
    have h1 : dist (f v) (f u) ≤ dist (f v) (f v') + dist (f v') (f u) := dist_triangle _ _ _
    have h2 : dist (f v') (f u) ≤ dist (f v') (f v) + dist (f v) (f u) := dist_triangle _ _ _
    rw [abs_le]
    constructor
    · linarith [dist_comm (f v') (f v)]
    · linarith
  unfold muN
  rw [mu_univ_split f δ H v', mu_univ_split f δ H v]
  unfold mu
  apply normalized_ratio_stability H (fun u => Real.exp (-δ * dist (f v) (f u)))
    (fun u => Real.exp (-δ * dist (f v') (f u))) _ hl
  · intro u; exact Real.exp_pos _
  · intro u; exact Real.exp_pos _
  · intro u; exact exp_kernel_ratio δ d _ _ hδ (htri u)
  · intro u
    have h := htri u
    rw [abs_sub_comm] at h
    exact exp_kernel_ratio δ d _ _ hδ h
  · exact hH
  · exact hHc

/-- **Normalized hidden-region stability.** Changing the hidden region changes the
normalized functional by at most the mass of the symmetric difference over the total mass. -/
theorem muN_stability_hidden (δ : ℝ) (hδ : 0 ≤ δ) (v : V) (H H' : Finset V) (k : ℕ)
    (hk : (H \ H').card + (H' \ H).card ≤ k) (hV : (Finset.univ : Finset V).Nonempty) :
    |muN f δ H v - muN f δ H' v| ≤ (k : ℝ) / mu f δ Finset.univ v := by
  have hpos := mu_pos_of_nonempty f δ Finset.univ v hV
  unfold muN
  rw [← sub_div, abs_div, abs_of_pos hpos]
  exact div_le_div_of_nonneg_right (mu_stability_II f δ hδ v H H' k hk) hpos.le

end

end Instantiation
