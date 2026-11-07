import ObserverGeometry.Instantiation.Stability
import Mathlib.Analysis.SpecificLimits.Basic

namespace Instantiation

noncomputable section

-- Sensitivity of `mu` (Proposition 2(a)-(c)).

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

-- Each hidden vertex contributes exactly `exp (-δ d(v,u))` to `mu`, with no interaction terms.
omit [Fintype V] in
theorem mu_influence (δ : ℝ) (v u : V) (H : Finset V) (hu : u ∉ H) :
    mu f δ (insert u H) v - mu f δ H v = Real.exp (-δ * dist (f v) (f u)) :=
  mu_stability_VII f δ v u H hu

-- The contribution of a hidden vertex is non-increasing in its distance.
theorem influence_antitone (δ : ℝ) (hδ : 0 ≤ δ) {d1 d2 : ℝ} (h : d1 ≤ d2) :
    Real.exp (-δ * d2) ≤ Real.exp (-δ * d1) := by
  apply Real.exp_le_exp.mpr
  nlinarith

-- Cardinality bounds: `|H| exp (-δ d_max) ≤ mu ≤ |H| exp (-δ d_min)`.
omit [Fintype V] [DecidableEq V] in
theorem mu_cardinality_bounds (δ : ℝ) (hδ : 0 ≤ δ) (v : V) (H : Finset V)
    (dmin dmax : ℝ) (hmin : ∀ u ∈ H, dmin ≤ dist (f v) (f u))
    (hmax : ∀ u ∈ H, dist (f v) (f u) ≤ dmax) :
    (H.card : ℝ) * Real.exp (-δ * dmax) ≤ mu f δ H v ∧
    mu f δ H v ≤ (H.card : ℝ) * Real.exp (-δ * dmin) := by
  unfold mu
  constructor
  · calc (H.card : ℝ) * Real.exp (-δ * dmax)
        = ∑ _u ∈ H, Real.exp (-δ * dmax) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ u ∈ H, Real.exp (-δ * dist (f v) (f u)) := by
          apply Finset.sum_le_sum
          intro u hu
          exact influence_antitone δ hδ (hmax u hu)
  · calc ∑ u ∈ H, Real.exp (-δ * dist (f v) (f u))
        ≤ ∑ _u ∈ H, Real.exp (-δ * dmin) := by
          apply Finset.sum_le_sum
          intro u hu
          exact influence_antitone δ hδ (hmin u hu)
      _ = (H.card : ℝ) * Real.exp (-δ * dmin) := by rw [Finset.sum_const, nsmul_eq_mul]

-- Limits in δ.

open Filter Topology

-- As δ → 0, mu recovers the hidden-node COUNT.
omit [Fintype V] [DecidableEq V] in
theorem mu_tendsto_card_as_delta_tendsto_zero (v : V) (H : Finset V) :
    Tendsto (fun δ => mu f δ H v) (𝓝 0) (𝓝 (H.card : ℝ)) := by
  unfold mu
  have h : Tendsto (fun δ => ∑ u ∈ H, Real.exp (-δ * dist (f v) (f u)))
      (𝓝 0) (𝓝 (∑ _u ∈ H, (1:ℝ))) := by
    apply tendsto_finsetSum
    intro u _
    have hlin : Continuous (fun δ : ℝ => -δ * dist (f v) (f u)) :=
      continuous_id.neg.mul continuous_const
    have hexp : Tendsto (fun δ : ℝ => Real.exp (-δ * dist (f v) (f u))) (𝓝 0)
        (𝓝 (Real.exp (-(0:ℝ) * dist (f v) (f u)))) :=
      (Real.continuous_exp.comp hlin).tendsto 0
    simpa using hexp
  simpa [Finset.sum_const, nsmul_eq_mul] using h

-- As δ → ∞, mu → 0, provided every hidden node is strictly positive
-- distance from v.
omit [Fintype V] [DecidableEq V] in
theorem mu_tendsto_zero_as_delta_tendsto_atTop (v : V) (H : Finset V)
    (hpos : ∀ u ∈ H, 0 < dist (f v) (f u)) :
    Tendsto (fun δ => mu f δ H v) atTop (𝓝 0) := by
  unfold mu
  have h : Tendsto (fun δ => ∑ u ∈ H, Real.exp (-δ * dist (f v) (f u)))
      atTop (𝓝 (∑ _u ∈ H, (0:ℝ))) := by
    apply tendsto_finsetSum
    intro u hu
    have hd := hpos u hu
    have hlin : Tendsto (fun δ : ℝ => -δ * dist (f v) (f u)) atTop atBot := by
      rw [tendsto_atBot]
      intro b
      filter_upwards [eventually_ge_atTop ((-b) / dist (f v) (f u))] with δ hδ
      have hkey : (-b / dist (f v) (f u)) * dist (f v) (f u) = -b := by
        field_simp
      have hge : δ * dist (f v) (f u) ≥ (-b / dist (f v) (f u)) * dist (f v) (f u) :=
        mul_le_mul_of_nonneg_right hδ (le_of_lt hd)
      rw [hkey] at hge
      linarith
    have hexp : Tendsto (fun δ : ℝ => Real.exp (-δ * dist (f v) (f u))) atTop (𝓝 0) :=
      Real.tendsto_exp_atBot.comp hlin
    simpa using hexp
  simpa using h

end

end Instantiation
