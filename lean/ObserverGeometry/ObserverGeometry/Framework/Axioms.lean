import Mathlib.Tactic

noncomputable section

-- Framework: observation processes, hidden regions, region functionals and the observer
-- descriptor. A finite vertex set V carries an observation process; budgets are indexed by a
-- monotone sequence of budget values `I : ℕ → ℝ`, so distinct indices may share a value.

variable (V : Type*) [Fintype V] [DecidableEq V]

-- An observation process (Definition 1): initialization, monotonicity in the budget value, and
-- eventual exhaustion (the process is ideal, `O v t = univ` for large t).
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

-- Region functional (Definition 3). `J v S` takes the observer `v` as an explicit basepoint,
-- so that functionals measuring distance to the observer, such as `mu`, are instances.
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

-- Observer descriptor (Definition 4): `Phi v = Σ_t P(t) * J v (H v t)`. The same `v` selects
-- the hidden region and serves as the basepoint of `J`.
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

-- Budget perturbation (Theorem 1(b)): `|Phi_P v - Phi_Q v| ≤ J v univ * Σ_t |P t - Q t|`
-- for budget distributions with a shared support.
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
