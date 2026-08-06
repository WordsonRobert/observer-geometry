import ObserverGeometry.Framework.Axioms

noncomputable section

-- Extension: Metric Locality and Isometry Invariance.
--
-- Brings in an ambient metric space (X, d_X) and an embedding
-- f : V -> X.
--
-- IMPORTANT CORRECTION: Locality is now BASEPOINT-RELATIVE. The
-- earlier version quantified over isometries between the induced
-- metric spaces (H1, d) and (H2, d) THEMSELVES -- i.e. pairwise
-- distances WITHIN H1, H2 -- and demanded J H1 = J H2. That axiom is
-- FALSE for any basepoint-dependent functional such as the concrete
-- exponential-weighted mu of Instantiation/Mu.lean, since mu depends
-- on distance TO THE OBSERVER, which is not intrinsic to H at all:
-- take H1 = {u1}, H2 = {u2} with d(v,u1) != d(v,u2). Any two
-- one-point metric spaces are trivially isometric, so the old axiom
-- would force mu(H1) = mu(H2), which is false whenever the two
-- points are genuinely at different distances from the observer.
--
-- The corrected axiom below quantifies over bijections between H1
-- (viewed from basepoint v1) and H2 (viewed from basepoint v2) that
-- preserve DISTANCE-TO-BASEPOINT pointwise -- equivalently, isometries
-- of H1 ∪ {v1} and H2 ∪ {v2} that send v1 to v2. This is exactly what
-- Instantiation/Canonicality.lean's radial-decomposition theorem
-- (mu depends only on the shell-count profile of distances to the
-- basepoint) already shows mu satisfies.

variable {V : Type*} [Fintype V] [DecidableEq V]
variable {X : Type*} [MetricSpace X] (f : V → X)

-- LOCALITY AXIOM (basepoint-relative form): for any two basepoints
-- v1, v2, any two subsets H1, H2, and any bijection e between them
-- (as subtypes) that preserves distance TO THE RESPECTIVE BASEPOINT
-- pointwise, J assigns the same value to (v1, H1) and (v2, H2).
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

-- ISOMETRY INVARIANCE (Corollary of the basepoint-relative Locality
-- axiom). If e : V ≃ V preserves f-distances EVERYWHERE (a genuine
-- global symmetry of the embedding), then applying e to BOTH the
-- basepoint and the hidden region leaves J invariant. Note both v
-- AND H move together under e -- this is the corrected statement;
-- previously J's basepoint-independence made this look like "J is
-- invariant under e(H) alone", which was not actually meaningful
-- once J genuinely depends on a basepoint.
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
