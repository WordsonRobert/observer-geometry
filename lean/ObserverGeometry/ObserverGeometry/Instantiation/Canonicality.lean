import ObserverGeometry.Instantiation.Sensitivity

namespace Instantiation

noncomputable section

-- Measure property and radial (shell) decomposition of `mu` (Proposition 2(d)).

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

-- `mu` satisfies the axioms of a finite measure on the subsets of V: non-negativity,
-- `mu ∅ = 0` and finite additivity on disjoint sets.
omit [Fintype V] in
theorem mu_is_finite_measure (δ : ℝ) (v : V) :
    (∀ H : Finset V, 0 ≤ mu f δ H v) ∧
    mu f δ (∅ : Finset V) v = 0 ∧
    (∀ A B : Finset V, Disjoint A B → mu f δ (A ∪ B) v = mu f δ A v + mu f δ B v) :=
  ⟨fun H => mu_nonneg f δ H v, mu_exhaustion f δ v, fun A B hAB => mu_additive f δ v A B hAB⟩

-- Radial decomposition: `mu` depends on a hidden region only through its shell counts, the
-- number of hidden vertices at each realized distance from the observer.

open Classical in
-- The distinct distance values realized by H (from v).
def shellValues (v : V) (H : Finset V) : Finset ℝ :=
  H.image (fun u => dist (f v) (f u))

open Classical in
-- The shell count: how many hidden nodes sit at exactly distance r.
def shellCount (v : V) (H : Finset V) (r : ℝ) : ℕ :=
  (H.filter (fun u => dist (f v) (f u) = r)).card

-- Radial decomposition theorem.
omit [Fintype V] [DecidableEq V] in
theorem mu_radial_decomposition (δ : ℝ) (v : V) (H : Finset V) :
    mu f δ H v = ∑ r ∈ shellValues f v H, (shellCount f v H r : ℝ) * Real.exp (-δ * r) := by
  unfold mu shellValues shellCount
  rw [← Finset.sum_fiberwise_of_maps_to
    (fun u hu => Finset.mem_image_of_mem (fun u => dist (f v) (f u)) hu)
    (f := fun u => Real.exp (-δ * dist (f v) (f u)))]
  apply Finset.sum_congr rfl
  intro r _
  have hconst : ∀ u ∈ H.filter (fun u => dist (f v) (f u) = r),
      Real.exp (-δ * dist (f v) (f u)) = Real.exp (-δ * r) := by
    intro u hu
    simp only [Finset.mem_filter] at hu
    rw [hu.2]
  rw [Finset.sum_congr rfl hconst, Finset.sum_const, nsmul_eq_mul]

end

end Instantiation
