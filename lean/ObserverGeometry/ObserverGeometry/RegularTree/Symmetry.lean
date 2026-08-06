import ObserverGeometry.RegularTree.DistanceDecomposition

namespace RegularTree

noncomputable section

-- Regular Tree: Theorem 16 (Same-Depth Symmetry) via an explicit
-- coordinate-swap tree automorphism.

variable (k D : ℕ)

open Classical in
def sigmaSwap (u v : TreeVertex k D) (i : ℕ) : choiceType k i ≃ choiceType k i :=
  if h : i < depth k D u ∧ i < depth k D v then
    Equiv.swap (u.2 ⟨i, h.1⟩) (v.2 ⟨i, h.2⟩)
  else Equiv.refl _

open Classical in
def swapMap (u v : TreeVertex k D) (w : TreeVertex k D) : TreeVertex k D :=
  ⟨w.1, fun j => sigmaSwap k D u v j.1 (w.2 j)⟩

open Classical in
theorem swapMap_involutive (u v w : TreeVertex k D) :
    swapMap k D u v (swapMap k D u v w) = w := by
  unfold swapMap
  cases w with
  | mk d p =>
    congr 1
    funext j
    have : sigmaSwap k D u v j.1 (sigmaSwap k D u v j.1 (p j)) = p j := by
      unfold sigmaSwap
      split
      · rw [Equiv.swap_apply_self]
      · rfl
    exact this

open Classical in
def swapAuto (u v : TreeVertex k D) : TreeVertex k D ≃ TreeVertex k D where
  toFun := swapMap k D u v
  invFun := swapMap k D u v
  left_inv := swapMap_involutive k D u v
  right_inv := swapMap_involutive k D u v

open Classical in
theorem swapAuto_maps_u_to_v (u v : TreeVertex k D) (hd : depth k D u = depth k D v) :
    swapAuto k D u v u = v := by
  change swapMap k D u v u = v
  unfold swapMap
  cases u with
  | mk du pu =>
    cases v with
    | mk dv pv =>
      simp only [depth] at hd
      have hdeq : du = dv := Fin.ext hd
      subst hdeq
      congr 1
      funext j
      unfold sigmaSwap
      simp only [depth]
      have hj : j.1 < du ∧ j.1 < du := ⟨j.2, j.2⟩
      rw [dif_pos hj]
      exact Equiv.swap_apply_left _ _

open Classical in
theorem swapAuto_maps_v_to_u (u v : TreeVertex k D) (hd : depth k D u = depth k D v) :
    swapAuto k D u v v = u := by
  change swapMap k D u v v = u
  unfold swapMap
  cases u with
  | mk du pu =>
    cases v with
    | mk dv pv =>
      simp only [depth] at hd
      have hdeq : du = dv := Fin.ext hd
      subst hdeq
      congr 1
      funext j
      unfold sigmaSwap
      simp only [depth]
      have hj : j.1 < du ∧ j.1 < du := ⟨j.2, j.2⟩
      rw [dif_pos hj]
      exact Equiv.swap_apply_right _ _

open Classical in
theorem swapAuto_preserves_agreesUpToP (u v w1 w2 : TreeVertex k D) (m : ℕ) :
    agreesUpToP k D w1 w2 m ↔ agreesUpToP k D (swapAuto k D u v w1) (swapAuto k D u v w2) m := by
  unfold agreesUpToP agreesUpTo
  constructor
  · rintro ⟨h1, h2, hagree⟩
    refine ⟨h1, h2, fun i => ?_⟩
    change sigmaSwap k D u v i.1 (w1.2 ⟨i.1, lt_of_lt_of_le i.2 h1⟩) =
      sigmaSwap k D u v i.1 (w2.2 ⟨i.1, lt_of_lt_of_le i.2 h2⟩)
    rw [hagree i]
  · rintro ⟨h1, h2, hagree⟩
    refine ⟨h1, h2, fun i => ?_⟩
    have hthis : sigmaSwap k D u v i.1 (w1.2 ⟨i.1, lt_of_lt_of_le i.2 h1⟩) =
        sigmaSwap k D u v i.1 (w2.2 ⟨i.1, lt_of_lt_of_le i.2 h2⟩) := hagree i
    exact (Equiv.injective (sigmaSwap k D u v i.1)) hthis

open Classical in
theorem swapAuto_preserves_depth (u v w : TreeVertex k D) :
    depth k D (swapAuto k D u v w) = depth k D w := rfl

open Classical in
theorem swapAuto_preserves_lcaDepth (u v w1 w2 : TreeVertex k D) :
    lcaDepth k D w1 w2 = lcaDepth k D (swapAuto k D u v w1) (swapAuto k D u v w2) := by
  unfold lcaDepth
  rw [swapAuto_preserves_depth, swapAuto_preserves_depth]
  congr 1
  funext m
  exact propext (swapAuto_preserves_agreesUpToP k D u v w1 w2 m)

open Classical in
theorem swapAuto_preserves_treeDist (u v w1 w2 : TreeVertex k D) :
    treeDist k D w1 w2 = treeDist k D (swapAuto k D u v w1) (swapAuto k D u v w2) := by
  unfold treeDist
  rw [swapAuto_preserves_depth, swapAuto_preserves_depth, swapAuto_preserves_lcaDepth k D u v w1 w2]

open Classical in
def muAtVertex (δ : ℝ) (s : TreeVertex k D) (R : ℕ) : ℝ :=
  ∑ w : TreeVertex k D, if R < treeDist k D s w then Real.exp (-δ * treeDist k D s w) else 0

open Classical in
theorem muAtVertex_eq_of_same_depth (δ : ℝ) (u v : TreeVertex k D)
    (hd : depth k D u = depth k D v) (R : ℕ) :
    muAtVertex k D δ u R = muAtVertex k D δ v R := by
  unfold muAtVertex
  rw [← Equiv.sum_comp (swapAuto k D u v)
    (fun w => if R < treeDist k D u w then Real.exp (-δ * treeDist k D u w) else 0)]
  apply Finset.sum_congr rfl
  intro w _
  have hkey : treeDist k D u (swapAuto k D u v w) = treeDist k D v w := by
    rw [swapAuto_preserves_treeDist k D u v v w, swapAuto_maps_v_to_u k D u v hd]
  rw [hkey]

open Classical in
theorem muAtVertex_depends_only_on_depth (δ : ℝ) (R : ℕ) (u v : TreeVertex k D)
    (hd : depth k D u = depth k D v) :
    muAtVertex k D δ u R = muAtVertex k D δ v R :=
  muAtVertex_eq_of_same_depth k D δ u v hd R

open Classical in
theorem muAtVertex_invariant_under_distance_preserving_equiv
    (f : TreeVertex k D ≃ TreeVertex k D)
    (hf : ∀ w1 w2, treeDist k D w1 w2 = treeDist k D (f w1) (f w2))
    (δ : ℝ) (s : TreeVertex k D) (R : ℕ) :
    muAtVertex k D δ (f s) R = muAtVertex k D δ s R := by
  unfold muAtVertex
  rw [← Equiv.sum_comp f
    (fun w => if R < treeDist k D (f s) w then Real.exp (-δ * treeDist k D (f s) w) else 0)]
  apply Finset.sum_congr rfl
  intro w _
  rw [← hf s w]

end

end RegularTree
