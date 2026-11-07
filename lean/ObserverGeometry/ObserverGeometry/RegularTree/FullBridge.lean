import ObserverGeometry.RegularTree.BridgeTheorem
import ObserverGeometry.Instantiation.Mu
import ObserverGeometry.Framework.Axioms

namespace RegularTree

noncomputable section

-- The Full Bridge: connects Framework/Instantiation's abstract `mu`
-- and `RGF.Phi` to the RegularTree development, closing the second
-- half of the gap (muAtVertex -> muRoot was proven in
-- BridgeTheorem.lean; this file proves mu -> muAtVertex, using
-- treeDist itself as a genuine MetricSpace on TreeVertex).

variable {k D : ℕ}

-- The tree vertices with `treeDist` form a metric space.
instance : MetricSpace (TreeVertex k D) where
  dist u v := (treeDist k D u v : ℝ)
  dist_self u := by
    have h := (treeDist_eq_zero_iff k D u u).mpr rfl
    exact_mod_cast h
  dist_comm u v := by
    exact_mod_cast treeDist_comm k D u v
  dist_triangle u v w := by
    exact_mod_cast treeDist_triangle k D u v w
  eq_of_dist_eq_zero {u v} h := by
    have h' : treeDist k D u v = 0 := by exact_mod_cast h
    exact (treeDist_eq_zero_iff k D u v).mp h'

def treeObs (k D : ℕ) : ObsProcess (TreeVertex k D) where
  I := Nat.cast
  I_mono := Nat.mono_cast
  O := fun v t => Finset.univ.filter (fun w => treeDist k D v w ≤ t)
  init := fun v => by
    ext w
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
    constructor
    · intro h
      exact ((treeDist_eq_zero_iff k D v w).mp (Nat.le_zero.mp h)).symm
    · intro h
      rw [h]
      exact le_of_eq (treeDist_eq_zero_iff k D v v |>.mpr rfl)
  mono := fun v n m h => by
    have hnm : n ≤ m := by exact_mod_cast h
    intro w hw
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
    omega
  eventual := fun v => by
    refine ⟨2 * D, fun t ht => ?_⟩
    ext w
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
    have hv : depth k D v ≤ D := Nat.lt_succ_iff.mp v.1.isLt
    have hw : depth k D w ≤ D := Nat.lt_succ_iff.mp w.1.isLt
    have hL1 := lcaDepth_le_left k D v w
    have hL2 := lcaDepth_le_right k D v w
    unfold treeDist
    omega

theorem mu_eq_muAtVertex (δ : ℝ) (s : TreeVertex k D) (R : ℕ) :
    Instantiation.mu (id : TreeVertex k D → TreeVertex k D) δ
      (Finset.univ.filter (fun w => R < treeDist k D s w)) s
    = muAtVertex k D δ s R := by
  unfold Instantiation.mu muAtVertex
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro a _
  simp only [id]
  rfl

-- Bridge (Theorem 2(b)): the framework descriptor `Phi`, instantiated on the tree with the
-- metric-ball observation process and `mu`, equals the closed form `muRoot` at the root.

theorem treeObs_H_eq (s : TreeVertex k D) (t : ℕ) :
    (treeObs k D).H s t = Finset.univ.filter (fun w => t < treeDist k D s w) := by
  unfold ObsProcess.H treeObs
  ext w
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and, Finset.mem_filter, not_le]

theorem J_treeObs_root (δ : ℝ) (R : ℕ) (hk : 1 ≤ k) :
    (Instantiation.muRGF (id : TreeVertex k D → TreeVertex k D) δ).J
      (root k D) ((treeObs k D).H (root k D) R) = muRoot k δ D R := by
  change Instantiation.mu id δ ((treeObs k D).H (root k D) R) (root k D) = muRoot k δ D R
  rw [treeObs_H_eq, mu_eq_muAtVertex]
  exact muAtVertex_root_eq_muRoot k D δ R hk

-- `Phi root = Σ_t P(t) * muRoot k δ D t` for every budget distribution P.
theorem Phi_root_eq (δ : ℝ) (P : BudgetDist) (hk : 1 ≤ k) :
    (Instantiation.muRGF (id : TreeVertex k D → TreeVertex k D) δ).Phi
      (treeObs k D) P (root k D) = ∑ t ∈ P.support, P.weight t * muRoot k δ D t := by
  unfold RGF.Phi
  apply Finset.sum_congr rfl
  intro t _
  rw [J_treeObs_root δ t hk]

-- BONUS (free, given the above): the framework descriptor Phi itself
-- depends only on depth, at the LEVEL of the actual instantiated
-- framework object -- not just muAtVertex in isolation.
theorem Phi_depends_only_on_depth (δ : ℝ) (P : BudgetDist) (u v : TreeVertex k D)
    (hd : depth k D u = depth k D v) :
    (Instantiation.muRGF (id : TreeVertex k D → TreeVertex k D) δ).Phi (treeObs k D) P u
      = (Instantiation.muRGF (id : TreeVertex k D → TreeVertex k D) δ).Phi (treeObs k D) P v := by
  unfold RGF.Phi
  apply Finset.sum_congr rfl
  intro t _
  congr 1
  change Instantiation.mu id δ ((treeObs k D).H u t) u
    = Instantiation.mu id δ ((treeObs k D).H v t) v
  rw [treeObs_H_eq, treeObs_H_eq, mu_eq_muAtVertex, mu_eq_muAtVertex]
  exact muAtVertex_depends_only_on_depth k D δ t u v hd

end

end RegularTree
