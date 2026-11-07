import ObserverGeometry.Framework.Axioms

noncomputable section

-- Metric locality and isometry invariance (Theorem 1(a)). The vertex set is embedded by
-- `f : V → X` into a metric space. Locality is relative to the basepoint: it quantifies over
-- bijections between hidden regions that preserve distance to the respective observer. (The
-- intrinsic form, over isometries of the hidden regions alone, fails for any functional that
-- depends on distance to the observer.)

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

-- Locality: for basepoints v1, v2, regions H1, H2 and a bijection between them that preserves
-- distance to the respective basepoint, J takes the same value on (v1, H1) and (v2, H2).
def Locality (Jf : RGF V) : Prop :=
  ∀ (v1 v2 : V) (H1 H2 : Finset V) (e : {x // x ∈ H1} ≃ {x // x ∈ H2}),
    (∀ x : {x // x ∈ H1}, dist (f v1) (f x.1) = dist (f v2) (f (e x).1)) →
    Jf.J v1 H1 = Jf.J v2 H2

-- A global bijection e : V ≃ V that preserves f-distances everywhere
-- induces, for any H, a bijection between (H, ·) and (H.image e, ·)
-- at the subtype level.
def globalToLocalEquiv (e : V ≃ V) (H : Finset V) :
    {x // x ∈ H} ≃ {x // x ∈ H.image e} where
  toFun := fun x => ⟨e x.1, Finset.mem_image_of_mem e x.2⟩
  invFun := fun y => ⟨e.symm y.1, by
    have hy : y.1 ∈ H.image e := y.2
    simp only [Finset.mem_image] at hy
    obtain ⟨x, hx, hex⟩ := hy
    have heq : e.symm y.1 = x := by rw [← hex, Equiv.symm_apply_apply]
    rw [heq]
    exact hx⟩
  left_inv := fun x => by simp
  right_inv := fun y => by simp

-- If `e : V ≃ V` preserves f-distances, applying `e` to both the basepoint and the hidden region
-- leaves J unchanged.
theorem isometry_invariance_of_locality (Jf : RGF V) (hLoc : Locality f Jf)
    (e : V ≃ V) (hpres : ∀ x y, dist (f (e x)) (f (e y)) = dist (f x) (f y))
    (v : V) (H : Finset V) :
    Jf.J (e v) (H.image e) = Jf.J v H := by
  exact (hLoc v (e v) H (H.image e) (globalToLocalEquiv e H)
    (fun x => (hpres v x.1).symm)).symm

-- FULL DESCRIPTOR ISOMETRY INVARIANCE. Extending the corollary above
-- to the observer descriptor Phi requires one further compatibility
-- hypothesis: e must also be compatible with the observation process
-- itself (e maps the observed region at v to the observed region at
-- e(v), for every budget).
theorem Phi_isometry_invariant (Jf : RGF V) (Obs : ObsProcess V) (P : BudgetDist)
    (hLoc : Locality f Jf)
    (e : V ≃ V) (hpres : ∀ x y, dist (f (e x)) (f (e y)) = dist (f x) (f y))
    (hcompat : ∀ v t, (Obs.O v t).image e = Obs.O (e v) t)
    (v : V) :
    Jf.Phi Obs P (e v) = Jf.Phi Obs P v := by
  unfold RGF.Phi
  apply Finset.sum_congr rfl
  intro t _
  have hH : (Obs.H v t).image e = Obs.H (e v) t := by
    unfold ObsProcess.H
    ext u
    simp only [Finset.mem_image, Finset.mem_sdiff, Finset.mem_univ, true_and]
    constructor
    · rintro ⟨w, hw, rfl⟩
      intro hcontra
      apply hw
      rw [← hcompat v t] at hcontra
      simp only [Finset.mem_image] at hcontra
      obtain ⟨w', hw', heq⟩ := hcontra
      have : w' = w := e.injective heq
      rwa [← this]
    · intro hu
      refine ⟨e.symm u, ?_, e.apply_symm_apply u⟩
      intro hcontra
      apply hu
      rw [← hcompat v t]
      simp only [Finset.mem_image]
      exact ⟨e.symm u, hcontra, e.apply_symm_apply u⟩
  congr 1
  rw [← hH]
  exact isometry_invariance_of_locality f Jf hLoc e hpres v (Obs.H v t)

end
