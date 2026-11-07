import ObserverGeometry.Instantiation.Mu
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Data.Fintype.BigOperators

/-!
# Coverage form of the expected remaining geometry

A single exploration trajectory has finitely many outcomes `ω : Ω`, with probabilities `p ω`
and visited vertex sets `S ω`. `K` independent trajectories have outcome `ω : Fin K → Ω`
with product probability, and observe `⋃ k, S (ω k)`. The expected remaining weight of the
unobserved vertices then factorizes over vertices:

  `E[Σ_{u unobserved} w u] = Σ_u w u * (Σ_{ω'} [u ∉ S ω'] p ω') ^ K`,

and, when `Σ p = 1`, the inner sum is `1 - p_u` with `p_u` the single-trajectory visit
probability of `u`. This is the identity behind Proposition 2; it holds for any weights `w`,
in particular `w u = exp (-δ d(v, u))`.
-/

namespace Instantiation

open Finset

variable {V Ω : Type*} [Fintype V] [DecidableEq V] [Fintype Ω]

/-- Probability that a single trajectory does not visit `u`. -/
def missProb (p : Ω → ℝ) (S : Ω → Finset V) (u : V) : ℝ :=
  ∑ ω, if u ∈ S ω then 0 else p ω

/-- Probability that a single trajectory visits `u`. -/
def visitProb (p : Ω → ℝ) (S : Ω → Finset V) (u : V) : ℝ :=
  ∑ ω, if u ∈ S ω then p ω else 0

omit [Fintype V] in
theorem missProb_eq_one_sub (p : Ω → ℝ) (S : Ω → Finset V) (hp : ∑ ω, p ω = 1) (u : V) :
    missProb p S u = 1 - visitProb p S u := by
  unfold missProb visitProb
  rw [← hp, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro ω _
  split_ifs <;> ring

/-- **Coverage identity.** Expected remaining weight after `K` i.i.d. trajectories. -/
theorem coverage_identity (K : ℕ) (p : Ω → ℝ) (S : Ω → Finset V) (w : V → ℝ) :
    ∑ ω : Fin K → Ω, (∏ k, p (ω k)) *
        ∑ u ∈ Finset.univ.filter (fun u => ∀ k, u ∉ S (ω k)), w u
      = ∑ u, w u * (missProb p S u) ^ K := by
  have h1 : ∀ ω : Fin K → Ω,
      ∑ u ∈ Finset.univ.filter (fun u => ∀ k, u ∉ S (ω k)), w u
        = ∑ u, w u * ∏ k, (if u ∈ S (ω k) then (0 : ℝ) else 1) := by
    intro ω
    rw [Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro u _
    by_cases h : ∀ k, u ∉ S (ω k)
    · rw [if_pos h, Finset.prod_eq_one (fun k _ => by rw [if_neg (h k)])]
      ring
    · push_neg at h
      obtain ⟨k, hk⟩ := h
      rw [if_neg (by push_neg; exact ⟨k, hk⟩),
        Finset.prod_eq_zero (Finset.mem_univ k) (by rw [if_pos hk])]
      ring
  simp_rw [h1, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro u _
  unfold missProb
  rw [← Finset.card_fin K, ← Finset.prod_const, Finset.card_fin]
  rw [Finset.prod_univ_sum]
  rw [Finset.mul_sum]
  simp only [Fintype.piFinset_univ]
  apply Finset.sum_congr rfl
  intro ω _
  have : (∏ k, (if u ∈ S (ω k) then (0 : ℝ) else p (ω k)))
      = (∏ k, p (ω k)) * ∏ k, (if u ∈ S (ω k) then (0 : ℝ) else 1) := by
    rw [← Finset.prod_mul_distrib]
    apply Finset.prod_congr rfl
    intro k _
    split_ifs <;> ring
  rw [this]
  ring

/-- **Coverage form of the expected normalized descriptor** (Proposition 2): for weights
`w u = exp (-δ d(v,u))`, total mass `μ(v,V)`, and a trajectory law with `Σ p = 1`. -/
theorem expected_muN_coverage (K : ℕ) (p : Ω → ℝ) (S : Ω → Finset V)
    (hp : ∑ ω, p ω = 1) {X : Type*} [MetricSpace X] (f : V → X) (δ : ℝ) (v : V) :
    (∑ ω : Fin K → Ω, (∏ k, p (ω k)) *
        mu f δ (Finset.univ.filter (fun u => ∀ k, u ∉ S (ω k))) v) / mu f δ Finset.univ v
      = (∑ u, Real.exp (-δ * dist (f v) (f u)) * (1 - visitProb p S u) ^ K)
          / mu f δ Finset.univ v := by
  congr 1
  unfold mu
  rw [coverage_identity K p S (fun u => Real.exp (-δ * dist (f v) (f u)))]
  apply Finset.sum_congr rfl
  intro u _
  rw [missProb_eq_one_sub p S hp u]

end Instantiation
