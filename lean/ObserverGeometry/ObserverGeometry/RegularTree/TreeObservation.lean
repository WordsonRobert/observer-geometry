import ObserverGeometry.RegularTree.Geometry

namespace RegularTree

noncomputable section

-- Regular Tree: The Bridge Theorem (TreeObservation).

variable (k D : ℕ)

instance : DecidableEq (TreeVertex k D) := Classical.decEq _

open Classical in
def root (k D : ℕ) : TreeVertex k D :=
  ⟨⟨0, Nat.succ_pos D⟩, fun i => absurd i.2 (Nat.not_lt_zero i.1)⟩

open Classical in
def O_root (k D : ℕ) (t : ℕ) : Finset (TreeVertex k D) :=
  Finset.univ.filter (fun v => depth k D v ≤ t)

open Classical in
def H_root (k D : ℕ) (t : ℕ) : Finset (TreeVertex k D) :=
  Finset.univ \ O_root k D t

open Classical in
theorem H_root_eq_depth_gt (t : ℕ) :
    H_root k D t = Finset.univ.filter (fun v => t < depth k D v) := by
  unfold H_root O_root
  ext v
  simp only [Finset.mem_sdiff, Finset.mem_filter, Finset.mem_univ, true_and, not_le]

open Classical in
theorem O_root_init : O_root k D 0 = {root k D} := by
  unfold O_root
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  obtain ⟨d, p⟩ := v
  constructor
  · intro hv
    have hd0 : (d:ℕ) = 0 := Nat.le_zero.mp hv
    have hdeq : d = (⟨0, Nat.succ_pos D⟩ : Fin (D + 1)) := Fin.ext hd0
    subst hdeq
    have hpeq : p = (fun i => absurd i.2 (Nat.not_lt_zero i.1)) := by
      funext i
      exact absurd i.2 (Nat.not_lt_zero i.1)
    rw [hpeq]
    rfl
  · intro hv
    rw [hv]
    unfold root depth
    rfl

open Classical in
theorem O_root_mono {t1 t2 : ℕ} (h : t1 ≤ t2) : O_root k D t1 ⊆ O_root k D t2 := by
  unfold O_root
  intro v hv
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hv ⊢
  omega

open Classical in
theorem O_root_eventual : O_root k D D = Finset.univ := by
  unfold O_root
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
  exact Nat.lt_succ_iff.mp v.1.isLt

end

end RegularTree
