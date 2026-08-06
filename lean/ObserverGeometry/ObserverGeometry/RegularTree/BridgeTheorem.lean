import ObserverGeometry.RegularTree.Symmetry
import ObserverGeometry.RegularTree.ExactDescriptor
import ObserverGeometry.RegularTree.TreeObservation

namespace RegularTree

noncomputable section

-- Regular Tree: The Missing Link.

theorem muAtVertex_root_eq_muRoot (k D : ℕ) (δ : ℝ) (R : ℕ) (hk : 1 ≤ k) :
    muAtVertex k D δ (root k D) R = muRoot k δ D R := by
  unfold muAtVertex muRoot
  have hdist : ∀ w : TreeVertex k D, treeDist k D (root k D) w = depth k D w :=
    treeDist_root_eq_depth k D
  simp_rw [hdist]
  rw [Fintype.sum_sigma (f := fun w : TreeVertex k D =>
    if R < depth k D w then Real.exp (-δ * depth k D w) else 0)]
  have hinner : ∀ d : Fin (D + 1),
      (∑ p : TreePath k d.1,
        if R < depth k D ⟨d, p⟩ then Real.exp (-δ * depth k D ⟨d, p⟩) else 0)
      = (Fintype.card (TreePath k d.1) : ℝ) * (if R < d.1 then Real.exp (-δ * (d.1:ℝ)) else 0) := by
    intro d
    have hconst : ∀ p : TreePath k d.1,
        (if R < depth k D ⟨d, p⟩ then Real.exp (-δ * depth k D ⟨d, p⟩) else 0)
        = (if R < d.1 then Real.exp (-δ * (d.1:ℝ)) else 0) := by
      intro p
      simp only [depth]
    rw [Finset.sum_congr rfl (fun p _ => hconst p)]
    rw [Finset.sum_const, nsmul_eq_mul]
    rfl
  simp_rw [hinner]
  rw [show (∑ d : Fin (D + 1), (Fintype.card (TreePath k d.1) : ℝ) *
      (if R < d.1 then Real.exp (-δ * (d.1:ℝ)) else 0))
    = ∑ d ∈ Finset.range (D + 1), (Fintype.card (TreePath k d) : ℝ) *
      (if R < d then Real.exp (-δ * (d:ℝ)) else 0) from
    Fin.sum_univ_eq_sum_range
      (fun d => (Fintype.card (TreePath k d) : ℝ) * (if R < d then Real.exp (-δ * (d:ℝ)) else 0))
      (D + 1)]
  have hpush : ∀ d ∈ Finset.range (D + 1),
      (Fintype.card (TreePath k d) : ℝ) * (if R < d then Real.exp (-δ * (d:ℝ)) else 0)
      = (if R < d then (Fintype.card (TreePath k d) : ℝ) * Real.exp (-δ * (d:ℝ)) else 0) := by
    intro d _
    split <;> ring
  rw [Finset.sum_congr rfl hpush, ← Finset.sum_filter]
  have hfilter : (Finset.range (D + 1)).filter (fun d => R < d) = Finset.Ico (R + 1) (D + 1) := by
    ext d
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
    omega
  rw [hfilter]
  apply Finset.sum_congr rfl
  intro r hr
  simp only [Finset.mem_Ico] at hr
  have hr1 : 1 ≤ r := by omega
  rw [shell_cardinality k r hk]
  rw [if_neg (by omega : ¬ r = 0)]
  push_cast [Nat.cast_sub hk]
  ring

end

end RegularTree
