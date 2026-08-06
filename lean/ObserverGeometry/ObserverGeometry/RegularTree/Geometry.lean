import Mathlib.Tactic

namespace RegularTree

noncomputable section

-- Regular Tree: Section 1, Geometry.

variable (k D : ℕ)

def choiceType (i : ℕ) : Type := Fin (if i = 0 then k else k - 1)

instance choiceTypeFintype (i : ℕ) : Fintype (choiceType k i) :=
  inferInstanceAs (Fintype (Fin (if i = 0 then k else k - 1)))

def TreePath (d : ℕ) : Type := (i : Fin d) → choiceType k i.1

instance TreePathFintype (d : ℕ) : Fintype (TreePath k d) :=
  inferInstanceAs (Fintype ((i : Fin d) → choiceType k i.1))

abbrev TreeVertex (k D : ℕ) : Type := Σ d : Fin (D + 1), TreePath k d.1

instance : Fintype (TreeVertex k D) :=
  inferInstanceAs (Fintype (Σ d : Fin (D + 1), TreePath k d.1))

def depth (v : TreeVertex k D) : ℕ := v.1.1

theorem depth_unique (v : TreeVertex k D) : ∃! d : ℕ, ∃ h : d ≤ D, ∃ p : TreePath k d,
    v = ⟨⟨d, Nat.lt_succ_of_le h⟩, p⟩ := by
  refine ⟨depth k D v, ⟨Nat.lt_succ_iff.mp v.1.isLt, v.2, ?_⟩, ?_⟩
  · cases v; rfl
  · rintro d ⟨_, _, rfl⟩
    rfl

theorem shell_cardinality (d : ℕ) (_hk : 1 ≤ k) :
    Fintype.card (TreePath k d) = if d = 0 then 1 else k * (k - 1) ^ (d - 1) := by
  change Fintype.card ((i : Fin d) → choiceType k i.1) = if d = 0 then 1 else k * (k - 1) ^ (d - 1)
  rw [Fintype.card_pi]
  rcases Nat.eq_zero_or_pos d with hd0 | hdpos
  · subst hd0; simp
  · rw [if_neg hdpos.ne']
    obtain ⟨e, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hdpos.ne'
    rw [Fin.prod_univ_succ]
    have h0 : Fintype.card (choiceType k (0 : Fin (e + 1)).1) = k := by
      change Fintype.card (Fin (if (0:ℕ) = 0 then k else k - 1)) = k
      simp
    have hrest : ∀ i : Fin e, Fintype.card (choiceType k (Fin.succ i).1) = k - 1 := by
      intro i
      change Fintype.card (Fin (if (Fin.succ i).1 = 0 then k else k - 1)) = k - 1
      simp [Fin.succ]
    rw [h0]
    rw [Finset.prod_congr rfl (fun i _ => hrest i)]
    rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    simp

end

end RegularTree
