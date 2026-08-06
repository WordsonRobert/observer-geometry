import ObserverGeometry.RegularTree.Geometry

namespace RegularTree

noncomputable section

-- Regular Tree: Theorem 3, Distance Decomposition (LCA).

variable (k D : ℕ)

open Classical in
def agreesUpTo (u v : TreeVertex k D) (m : ℕ)
    (hu : m ≤ depth k D u) (hv : m ≤ depth k D v) : Prop :=
  ∀ i : Fin m, u.2 ⟨i.1, lt_of_lt_of_le i.2 hu⟩ = v.2 ⟨i.1, lt_of_lt_of_le i.2 hv⟩

open Classical in
def agreesUpToP (u v : TreeVertex k D) (m : ℕ) : Prop :=
  ∃ hu : m ≤ depth k D u, ∃ hv : m ≤ depth k D v, agreesUpTo k D u v m hu hv

open Classical in
def lcaDepth (u v : TreeVertex k D) : ℕ :=
  Nat.findGreatest (agreesUpToP k D u v) (min (depth k D u) (depth k D v))

open Classical in
theorem lcaDepth_le_left (u v : TreeVertex k D) :
    lcaDepth k D u v ≤ depth k D u := by
  unfold lcaDepth
  exact le_trans (Nat.findGreatest_le _) (min_le_left _ _)

open Classical in
theorem lcaDepth_le_right (u v : TreeVertex k D) :
    lcaDepth k D u v ≤ depth k D v := by
  unfold lcaDepth
  exact le_trans (Nat.findGreatest_le _) (min_le_right _ _)

open Classical in
def treeDist (u v : TreeVertex k D) : ℕ :=
  (depth k D u - lcaDepth k D u v) + (depth k D v - lcaDepth k D u v)

open Classical in
theorem treeDist_eq_decomposition (u v : TreeVertex k D) :
    treeDist k D u v =
      (depth k D u - lcaDepth k D u v) + (depth k D v - lcaDepth k D u v) := rfl

open Classical in
theorem lcaDepth_comm (u v : TreeVertex k D) : lcaDepth k D u v = lcaDepth k D v u := by
  have hPsymm : ∀ m, agreesUpToP k D u v m ↔ agreesUpToP k D v u m := by
    intro m
    unfold agreesUpToP agreesUpTo
    constructor
    · rintro ⟨hu, hv, hagree⟩
      exact ⟨hv, hu, fun i => (hagree i).symm⟩
    · rintro ⟨hv, hu, hagree⟩
      exact ⟨hu, hv, fun i => (hagree i).symm⟩
  unfold lcaDepth
  rw [min_comm (depth k D u) (depth k D v)]
  congr 1
  funext m
  exact propext (hPsymm m)

open Classical in
theorem treeDist_comm (u v : TreeVertex k D) : treeDist k D u v = treeDist k D v u := by
  unfold treeDist
  rw [lcaDepth_comm]
  ring

open Classical in
theorem lcaDepth_root_eq_zero (v : TreeVertex k D) :
    lcaDepth k D ⟨⟨0, Nat.succ_pos D⟩, fun i => absurd i.2 (Nat.not_lt_zero i.1)⟩ v = 0 := by
  unfold lcaDepth
  apply le_antisymm
  · apply Nat.findGreatest_le
  · apply Nat.le_findGreatest (Nat.zero_le _)
    exact ⟨Nat.zero_le _, Nat.zero_le _, fun i => absurd i.2 (Nat.not_lt_zero i.1)⟩

open Classical in
theorem treeDist_root_eq_depth (v : TreeVertex k D) :
    treeDist k D ⟨⟨0, Nat.succ_pos D⟩, fun i => absurd i.2 (Nat.not_lt_zero i.1)⟩ v
      = depth k D v := by
  unfold treeDist
  rw [lcaDepth_root_eq_zero]
  simp [depth]

-- Towards treeDist being a genuine metric.

open Classical in
theorem agreesUpToP_downward (u v : TreeVertex k D) {m m' : ℕ} (hm'm : m' ≤ m)
    (h : agreesUpToP k D u v m) : agreesUpToP k D u v m' := by
  obtain ⟨hu, hv, hagree⟩ := h
  refine ⟨le_trans hm'm hu, le_trans hm'm hv, ?_⟩
  intro i
  exact hagree ⟨i.1, lt_of_lt_of_le i.2 hm'm⟩

open Classical in
theorem agreesUpToP_zero (u v : TreeVertex k D) : agreesUpToP k D u v 0 :=
  ⟨Nat.zero_le _, Nat.zero_le _, fun i => absurd i.2 (Nat.not_lt_zero i.1)⟩

-- lcaDepth actually satisfies the property that makes it "the LCA
-- depth" rather than merely a well-defined number.
open Classical in
theorem lcaDepth_agrees (u v : TreeVertex k D) :
    agreesUpToP k D u v (lcaDepth k D u v) := by
  unfold lcaDepth
  exact Nat.findGreatest_spec (Nat.zero_le _) (agreesUpToP_zero k D u v)

open Classical in
theorem lcaDepth_self_eq_depth (u : TreeVertex k D) :
    lcaDepth k D u u = depth k D u := by
  unfold lcaDepth
  apply le_antisymm
  · exact le_trans (Nat.findGreatest_le _) (min_le_left _ _)
  · apply Nat.le_findGreatest
    · exact le_of_eq (min_self (depth k D u)).symm
    · exact ⟨le_refl _, le_refl _, fun i => rfl⟩

-- treeDist(u,v) = 0 ↔ u = v.
open Classical in
theorem treeDist_eq_zero_iff (u v : TreeVertex k D) : treeDist k D u v = 0 ↔ u = v := by
  constructor
  · intro h
    unfold treeDist at h
    have hL := lcaDepth_le_left k D u v
    have hR := lcaDepth_le_right k D u v
    have he1 : depth k D u = lcaDepth k D u v := by omega
    have hdepth_eq : depth k D u = depth k D v := by omega
    obtain ⟨_, _, hagree⟩ := lcaDepth_agrees k D u v
    cases u with
    | mk du pu =>
      cases v with
      | mk dv pv =>
        simp only [depth] at hdepth_eq he1
        have hdu_eq : du = dv := Fin.ext hdepth_eq
        subst hdu_eq
        congr 1
        funext i
        have hi_lt : i.1 < lcaDepth k D ⟨du, pu⟩ ⟨du, pv⟩ := by
          rw [← he1]; exact i.2
        exact hagree ⟨i.1, hi_lt⟩
  · intro h
    subst h
    unfold treeDist
    rw [lcaDepth_self_eq_depth]
    simp

-- THE FOUR-POINT CONDITION (key structural fact about trees). Among
-- the three pairwise LCA-depths L(u,v), L(v,w), L(u,w), the third is
-- always at least the min of the other two -- provable directly from
-- downward-closure of agreement plus transitivity: if u,v agree on
-- their first m symbols and v,w agree on their first m symbols, then
-- u,w agree on their first m symbols too.
open Classical in
theorem lcaDepth_ge_min (u v w : TreeVertex k D) :
    min (lcaDepth k D u v) (lcaDepth k D v w) ≤ lcaDepth k D u w := by
  have huv : agreesUpToP k D u v (min (lcaDepth k D u v) (lcaDepth k D v w)) :=
    agreesUpToP_downward k D u v (min_le_left _ _) (lcaDepth_agrees k D u v)
  have hvw : agreesUpToP k D v w (min (lcaDepth k D u v) (lcaDepth k D v w)) :=
    agreesUpToP_downward k D v w (min_le_right _ _) (lcaDepth_agrees k D v w)
  obtain ⟨hu, _, hagree_uv⟩ := huv
  obtain ⟨_, hw, hagree_vw⟩ := hvw
  have huw : agreesUpToP k D u w (min (lcaDepth k D u v) (lcaDepth k D v w)) :=
    ⟨hu, hw, fun i => (hagree_uv i).trans (hagree_vw i)⟩
  unfold lcaDepth
  exact Nat.le_findGreatest (le_min hu hw) huw

-- THE TRIANGLE INEQUALITY. Follows purely from lcaDepth_ge_min
-- together with the existing bounds L(a,b) ≤ depth a, L(a,b) ≤ depth
-- b -- pure linear arithmetic once those facts are assembled, closed
-- directly by omega (which natively understands min and truncated
-- ℕ subtraction).
theorem treeDist_triangle (u v w : TreeVertex k D) :
    treeDist k D u w ≤ treeDist k D u v + treeDist k D v w := by
  have hmin := lcaDepth_ge_min k D u v w
  have h1 := lcaDepth_le_left k D u v
  have h2 := lcaDepth_le_right k D u v
  have h3 := lcaDepth_le_left k D v w
  have h4 := lcaDepth_le_right k D v w
  have h5 := lcaDepth_le_left k D u w
  have h6 := lcaDepth_le_right k D u w
  unfold treeDist
  omega

end

end RegularTree
