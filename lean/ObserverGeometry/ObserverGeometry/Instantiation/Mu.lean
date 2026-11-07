import ObserverGeometry.Framework.Axioms
import ObserverGeometry.Framework.MetricLocality
import Mathlib.Analysis.SpecialFunctions.Exp

namespace Instantiation

noncomputable section

-- The exponential region functional `mu δ H v = Σ_{u ∈ H} exp (-δ d(f v, f u))` (Eq. 2). For a
-- fixed hidden region H, `mu δ H` is a function of the observer v.

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

def mu (δ : ℝ) (H : Finset V) (v : V) : ℝ :=
  ∑ u ∈ H, Real.exp (-δ * dist (f v) (f u))

omit [Fintype V] [DecidableEq V] in
theorem mu_nonneg (δ : ℝ) (H : Finset V) (v : V) : 0 ≤ mu f δ H v :=
  Finset.sum_nonneg (fun _ _ => le_of_lt (Real.exp_pos _))

omit [Fintype V] [DecidableEq V] in
theorem mu_mono (δ : ℝ) (v : V) (H1 H2 : Finset V) (h : H1 ⊆ H2) :
    mu f δ H1 v ≤ mu f δ H2 v :=
  Finset.sum_le_sum_of_subset_of_nonneg h (fun _ _ _ => le_of_lt (Real.exp_pos _))

omit [Fintype V] [DecidableEq V] in
theorem mu_exhaustion (δ : ℝ) (v : V) : mu f δ ∅ v = 0 := by
  unfold mu; simp

-- `mu` as a region functional (`RGF` instance).
def muRGF (δ : ℝ) : RGF V where
  J := fun v H => mu f δ H v
  nonneg := fun v H => mu_nonneg f δ H v
  mono := fun v H1 H2 h => mu_mono f δ v H1 H2 h
  exhaustion := fun v => mu_exhaustion f δ v

omit [Fintype V] in
set_option linter.unusedDecidableInType false in
theorem mu_strict_mono (δ : ℝ) (v : V) (H1 H2 : Finset V) (h : H1 ⊂ H2) :
    mu f δ H1 v < mu f δ H2 v := by
  obtain ⟨hsub, hne⟩ := h
  obtain ⟨u, hu2, hu1⟩ := Finset.exists_of_ssubset ⟨hsub, hne⟩
  unfold mu
  have hins : mu f δ (insert u H1) v = mu f δ H1 v + Real.exp (-δ * dist (f v) (f u)) := by
    unfold mu
    rw [Finset.sum_insert hu1]
    ring
  calc ∑ x ∈ H1, Real.exp (-δ * dist (f v) (f x))
      < ∑ x ∈ H1, Real.exp (-δ * dist (f v) (f x)) + Real.exp (-δ * dist (f v) (f u)) := by
        linarith [Real.exp_pos (-δ * dist (f v) (f u))]
    _ = mu f δ (insert u H1) v := hins.symm
    _ ≤ mu f δ H2 v := mu_mono f δ v _ _ (Finset.insert_subset hu2 hsub)

omit [Fintype V] in
theorem mu_additive (δ : ℝ) (v : V) (A B : Finset V) (hdisj : Disjoint A B) :
    mu f δ (A ∪ B) v = mu f δ A v + mu f δ B v := by
  unfold mu
  exact Finset.sum_union hdisj

omit [Fintype V] in
theorem mu_inclusion_exclusion (δ : ℝ) (v : V) (A B : Finset V) :
    mu f δ (A ∪ B) v = mu f δ A v + mu f δ B v - mu f δ (A ∩ B) v := by
  have hAeq : A = (A \ B) ∪ (A ∩ B) := by
    ext x; simp only [Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter]; tauto
  have hUeq : A ∪ B = (A \ B) ∪ B := by
    ext x; simp only [Finset.mem_union, Finset.mem_sdiff]; tauto
  have hdisj1 : Disjoint (A \ B) (A ∩ B) := by
    rw [Finset.disjoint_left]
    intro x hx hx2
    simp only [Finset.mem_sdiff] at hx
    simp only [Finset.mem_inter] at hx2
    exact hx.2 hx2.2
  have hdisj2 : Disjoint (A \ B) B := by
    rw [Finset.disjoint_left]
    intro x hx hxB
    simp only [Finset.mem_sdiff] at hx
    exact hx.2 hxB
  have hA : mu f δ A v = mu f δ (A \ B) v + mu f δ (A ∩ B) v := by
    conv_lhs => rw [hAeq]
    exact mu_additive f δ v (A \ B) (A ∩ B) hdisj1
  have hU : mu f δ (A ∪ B) v = mu f δ (A \ B) v + mu f δ B v := by
    conv_lhs => rw [hUeq]
    exact mu_additive f δ v (A \ B) B hdisj2
  rw [hU, hA]
  ring

-- `mu` satisfies the basepoint-relative locality axiom: reindex the defining sum along the
-- distance-preserving bijection.
theorem mu_locality (δ : ℝ) : Locality f (muRGF f δ) := by
  intro v1 v2 H1 H2 e hpres
  change mu f δ H1 v1 = mu f δ H2 v2
  unfold mu
  apply Finset.sum_bij' (fun a ha => (e ⟨a, ha⟩).1) (fun b hb => (e.symm ⟨b, hb⟩).1)
  · intro a ha
    exact congrArg Subtype.val (e.symm_apply_apply ⟨a, ha⟩)
  · intro b hb
    exact congrArg Subtype.val (e.apply_symm_apply ⟨b, hb⟩)
  · intro a ha
    rw [hpres ⟨a, ha⟩]

end

end Instantiation
