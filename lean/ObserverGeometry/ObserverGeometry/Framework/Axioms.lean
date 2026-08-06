import Mathlib.Tactic

noncomputable section

-- Framework: Observer Geometry (Option B indexing).
--
-- G = (V, E) a finite graph, embedded via f : V -> X into a metric
-- space (X, d_X). A node v in V is called an observer.
--
-- Observation budgets are NOT the raw index n. Instead, `I : Nat ->
-- Real` is an arbitrary nondecreasing sequence of budget VALUES --
-- I n is the n-th budget in the sequence, and different indices n, m
-- may share the same budget value (I n = I m is allowed). The
-- observed region O depends only on the budget value, not on which
-- index produced it.

variable (V : Type*) [Fintype V] [DecidableEq V]

-- NOTE: an earlier draft of this structure included a `welldef`
-- field (O depends only on the budget VALUE I n, not on which index
-- n produced it). It was never used by any downstream proof in this
-- project -- every theorem here only ever needs `mono`, `init`, and
-- `eventual` -- so it has been dropped rather than kept as unused
-- machinery. The `I : Nat -> Real` budget-value sequence itself is
-- still genuinely used (I_mono drives H_comparable).
structure ObsProcess where
  I : ℕ → ℝ
  I_mono : Monotone I
  O : V → ℕ → Finset V
  init : ∀ v, O v 0 = {v}
  mono : ∀ v, ∀ n m, I n ≤ I m → O v n ⊆ O v m
  eventual : ∀ v, ∃ K, ∀ t, t ≥ K → O v t = Finset.univ

namespace ObsProcess

variable {V} (Obs : ObsProcess V)

def H (v : V) (t : ℕ) : Finset V := Finset.univ \ Obs.O v t

theorem H_subset_of_le (v : V) {n m : ℕ} (h : Obs.I n ≤ Obs.I m) :
    Obs.H v m ⊆ Obs.H v n := by
  unfold H
  intro u hu
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and] at hu ⊢
  intro hcontra
  exact hu (Obs.mono v n m h hcontra)

theorem H_comparable (v : V) (n m : ℕ) :
    Obs.H v n ⊆ Obs.H v m ∨ Obs.H v m ⊆ Obs.H v n := by
  rcases le_total n m with h | h
  · right; exact Obs.H_subset_of_le v (Obs.I_mono h)
  · left; exact Obs.H_subset_of_le v (Obs.I_mono h)

end ObsProcess

-- RGF (Remaining Geometry Functional).
--
-- IMPORTANT: J now takes an explicit BASEPOINT argument, J : V ->
-- Finset V -> Real. This is a deliberate correction: the previous
-- signature J : Finset V -> Real had no way to express that a
-- concrete remaining-mass functional (e.g. the exponential-weighted
-- mu of Instantiation/Mu.lean) genuinely depends on WHICH vertex is
-- doing the observing, not just on which set is hidden. Without this
-- basepoint slot, Phi (e v) silently reused the basepoint-independent
-- J at the WRONG basepoint whenever J was instantiated from a
-- basepoint-dependent concrete functional -- a real semantic bug,
-- not merely a missing generality.
structure RGF (V : Type*) [Fintype V] [DecidableEq V] where
  J : V → Finset V → ℝ
  nonneg : ∀ v H, 0 ≤ J v H
  mono : ∀ (v : V) (H1 H2 : Finset V), H1 ⊆ H2 → J v H1 ≤ J v H2
  exhaustion : ∀ v, J v ∅ = 0

structure BudgetDist where
  support : Finset ℕ
  weight : ℕ → ℝ
  weight_nonneg : ∀ t ∈ support, 0 ≤ weight t
  weight_sum_one : ∑ t ∈ support, weight t = 1

namespace RGF

variable {V : Type*} [Fintype V] [DecidableEq V]
variable (Jf : RGF V) (Obs : ObsProcess V) (P : BudgetDist)

theorem hidden_order_reflection
    (hstrict : ∀ v, ∀ H1 H2 : Finset V, H1 ⊂ H2 → Jf.J v H1 < Jf.J v H2)
    (v : V) (n m : ℕ) :
    Obs.H v n ⊆ Obs.H v m ↔ Jf.J v (Obs.H v n) ≤ Jf.J v (Obs.H v m) := by
  constructor
  · intro h
    exact Jf.mono v _ _ h
  · intro hJ
    rcases Obs.H_comparable v n m with hcomp | hcomp
    · exact hcomp
    · rcases eq_or_ssubset_of_subset hcomp with heq | hstrict_sub
      · rw [heq]
      · exfalso
        have := hstrict v _ _ hstrict_sub
        linarith

-- THE OBSERVER DESCRIPTOR: Phi(v) = Σ_I P(I) * J(v, H_I(v)). Note the
-- SAME v is used both to pick the hidden set AND as J's basepoint --
-- this is precisely the fix. Previously, instantiating Phi with a
-- basepoint-fixed J meant Phi(e v) still measured distances from the
-- ORIGINAL basepoint even though the hidden set moved to e(v).
def Phi (v : V) : ℝ :=
  ∑ t ∈ P.support, P.weight t * Jf.J v (Obs.H v t)

theorem Phi_nonneg (v : V) : 0 ≤ Jf.Phi Obs P v := by
  apply Finset.sum_nonneg
  intro t ht
  exact mul_nonneg (P.weight_nonneg t ht) (Jf.nonneg v _)

theorem Phi_le_J_univ (v : V) : Jf.Phi Obs P v ≤ Jf.J v Finset.univ := by
  unfold Phi
  have hbound : ∀ t, Jf.J v (Obs.H v t) ≤ Jf.J v Finset.univ := by
    intro t
    exact Jf.mono v _ _ (Finset.subset_univ _)
  calc ∑ t ∈ P.support, P.weight t * Jf.J v (Obs.H v t)
      ≤ ∑ t ∈ P.support, P.weight t * Jf.J v Finset.univ := by
        apply Finset.sum_le_sum
        intro t ht
        exact mul_le_mul_of_nonneg_left (hbound t) (P.weight_nonneg t ht)
    _ = (∑ t ∈ P.support, P.weight t) * Jf.J v Finset.univ := by rw [Finset.sum_mul]
    _ = Jf.J v Finset.univ := by rw [P.weight_sum_one, one_mul]

theorem J_antitone (v : V) {n m : ℕ} (h : Obs.I n ≤ Obs.I m) :
    Jf.J v (Obs.H v m) ≤ Jf.J v (Obs.H v n) :=
  Jf.mono v _ _ (Obs.H_subset_of_le v h)

theorem Phi_le_sup (v : V) (hne : P.support.Nonempty) :
    Jf.Phi Obs P v ≤ P.support.sup' hne (fun t => Jf.J v (Obs.H v t)) := by
  unfold Phi
  have hbound : ∀ t ∈ P.support,
      Jf.J v (Obs.H v t) ≤ P.support.sup' hne (fun t => Jf.J v (Obs.H v t)) := by
    intro t ht
    exact Finset.le_sup' (fun t => Jf.J v (Obs.H v t)) ht
  calc ∑ t ∈ P.support, P.weight t * Jf.J v (Obs.H v t)
      ≤ ∑ t ∈ P.support, P.weight t * P.support.sup' hne (fun t => Jf.J v (Obs.H v t)) := by
        apply Finset.sum_le_sum
        intro t ht
        exact mul_le_mul_of_nonneg_left (hbound t ht) (P.weight_nonneg t ht)
    _ = (∑ t ∈ P.support, P.weight t) * P.support.sup' hne (fun t => Jf.J v (Obs.H v t)) := by
        rw [Finset.sum_mul]
    _ = P.support.sup' hne (fun t => Jf.J v (Obs.H v t)) := by rw [P.weight_sum_one, one_mul]

theorem Phi_ge_inf (v : V) (hne : P.support.Nonempty) :
    P.support.inf' hne (fun t => Jf.J v (Obs.H v t)) ≤ Jf.Phi Obs P v := by
  unfold Phi
  have hbound : ∀ t ∈ P.support,
      P.support.inf' hne (fun t => Jf.J v (Obs.H v t)) ≤ Jf.J v (Obs.H v t) := by
    intro t ht
    exact Finset.inf'_le (fun t => Jf.J v (Obs.H v t)) ht
  calc P.support.inf' hne (fun t => Jf.J v (Obs.H v t))
      = (∑ t ∈ P.support, P.weight t) * P.support.inf' hne (fun t => Jf.J v (Obs.H v t)) := by
        rw [P.weight_sum_one, one_mul]
    _ = ∑ t ∈ P.support, P.weight t * P.support.inf' hne (fun t => Jf.J v (Obs.H v t)) := by
        rw [Finset.sum_mul]
    _ ≤ ∑ t ∈ P.support, P.weight t * Jf.J v (Obs.H v t) := by
        apply Finset.sum_le_sum
        intro t ht
        exact mul_le_mul_of_nonneg_left (hbound t ht) (P.weight_nonneg t ht)

theorem J_eventually_zero (v : V) :
    ∃ K, ∀ t ≥ K, Jf.J v (Obs.H v t) = 0 := by
  obtain ⟨K, hK⟩ := Obs.eventual v
  refine ⟨K, fun t ht => ?_⟩
  have hempty : Obs.H v t = ∅ := by
    unfold ObsProcess.H
    rw [hK t ht]
    simp
  rw [hempty, Jf.exhaustion]

theorem Phi_eq_zero_iff (v : V)
    (hstrict : ∀ v, ∀ H : Finset V, H ≠ ∅ → 0 < Jf.J v H) :
    Jf.Phi Obs P v = 0 ↔ ∀ t ∈ P.support, P.weight t > 0 → Obs.H v t = ∅ := by
  constructor
  · intro hPhi t ht hwt
    by_contra hne
    have hJpos : 0 < Jf.J v (Obs.H v t) := hstrict v _ hne
    have hterm_pos : 0 < P.weight t * Jf.J v (Obs.H v t) := mul_pos hwt hJpos
    have hall_nonneg : ∀ t' ∈ P.support, 0 ≤ P.weight t' * Jf.J v (Obs.H v t') := by
      intro t' ht'
      exact mul_nonneg (P.weight_nonneg t' ht') (Jf.nonneg v _)
    have : P.weight t * Jf.J v (Obs.H v t) ≤ Jf.Phi Obs P v :=
      Finset.single_le_sum hall_nonneg ht
    linarith [hPhi ▸ this]
  · intro hall
    unfold Phi
    apply Finset.sum_eq_zero
    intro t ht
    rcases eq_or_lt_of_le (P.weight_nonneg t ht) with hw0 | hwpos
    · rw [← hw0]; ring
    · rw [hall t ht hwpos, Jf.exhaustion]; ring

theorem hidden_mono (v : V) (t1 t2 : ℕ) (h : Obs.H v t1 ⊆ Obs.H v t2) :
    Jf.J v (Obs.H v t1) ≤ Jf.J v (Obs.H v t2) :=
  Jf.mono v _ _ h

-- STABILITY IV (framework-level). The bound constant J(v, V) is now
-- correctly BASEPOINT-DEPENDENT, matching the actual behavior of the
-- concrete mu instantiation (its "J(V)" is genuinely observer-
-- dependent, as documented in the experimental panels).
theorem Phi_stability_budget (P Q : BudgetDist) (hsupp : P.support = Q.support)
    (v : V) :
    |Jf.Phi Obs P v - Jf.Phi Obs Q v| ≤
      Jf.J v Finset.univ * ∑ t ∈ P.support, |P.weight t - Q.weight t| := by
  unfold Phi
  rw [show Q.support = P.support from hsupp.symm]
  calc |∑ t ∈ P.support, P.weight t * Jf.J v (Obs.H v t) -
        ∑ t ∈ P.support, Q.weight t * Jf.J v (Obs.H v t)|
      = |∑ t ∈ P.support, (P.weight t - Q.weight t) * Jf.J v (Obs.H v t)| := by
        rw [← Finset.sum_sub_distrib]
        congr 1
        apply Finset.sum_congr rfl
        intro t _
        ring
    _ ≤ ∑ t ∈ P.support, |(P.weight t - Q.weight t) * Jf.J v (Obs.H v t)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ t ∈ P.support, |P.weight t - Q.weight t| * Jf.J v (Obs.H v t) := by
        apply Finset.sum_congr rfl
        intro t _
        rw [abs_mul, abs_of_nonneg (Jf.nonneg v (Obs.H v t))]
    _ ≤ ∑ t ∈ P.support, |P.weight t - Q.weight t| * Jf.J v Finset.univ := by
        apply Finset.sum_le_sum
        intro t _
        exact mul_le_mul_of_nonneg_left
          (Jf.mono v _ _ (Finset.subset_univ _)) (abs_nonneg _)
    _ = Jf.J v Finset.univ * ∑ t ∈ P.support, |P.weight t - Q.weight t| := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro t _
        ring

end RGF

end
