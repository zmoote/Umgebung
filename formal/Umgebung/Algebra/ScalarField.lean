/-!
# Umgebung.Algebra.ScalarField

Self-contained algebraic foundations for Umgebung's Lean 4 formal proofs:
- `ScalarField α`: A commutative field of characteristic `≠ 2` capturing the
  exact algebraic laws shared by the real numbers ℝ and rational numbers ℚ
  without requiring a multi-gigabyte external Mathlib cache.
- `Vec3 α`: 3D Cartesian vector space mirroring `umgebung::scene::Vec3d`
  (`include/umgebung/scene/UniverseScene.hpp`), complete with vector addition,
  negation, scalar multiplication, dot product, cross product, and proven
  algebraic identities.
-/

namespace Umgebung.Algebra

/--
`ScalarField α` specifies a commutative field of characteristic `≠ 2` with
explicit algebraic laws matching standard real/rational arithmetic used in
theoretical physics derivations and numerical analysis.
-/
class ScalarField (α : Type u) where
  zero : α
  one : α
  two : α
  three : α
  four : α
  half : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  neg : α → α
  add_assoc : ∀ a b c : α, add (add a b) c = add a (add b c)
  add_comm : ∀ a b : α, add a b = add b a
  zero_add : ∀ a : α, add zero a = a
  add_zero : ∀ a : α, add a zero = a
  add_neg_cancel : ∀ a : α, add a (neg a) = zero
  neg_add_cancel : ∀ a : α, add (neg a) a = zero
  sub_eq_add_neg : ∀ a b : α, sub a b = add a (neg b)
  mul_assoc : ∀ a b c : α, mul (mul a b) c = mul a (mul b c)
  mul_comm : ∀ a b : α, mul a b = mul b a
  one_mul : ∀ a : α, mul one a = a
  mul_one : ∀ a : α, mul a one = a
  zero_mul : ∀ a : α, mul zero a = zero
  mul_zero : ∀ a : α, mul a zero = zero
  left_distrib : ∀ a b c : α, mul a (add b c) = add (mul a b) (mul a c)
  right_distrib : ∀ a b c : α, mul (add a b) c = add (mul a c) (mul b c)
  neg_mul : ∀ a b : α, mul (neg a) b = neg (mul a b)
  mul_neg : ∀ a b : α, mul a (neg b) = neg (mul a b)
  neg_neg : ∀ a : α, neg (neg a) = a
  neg_zero : neg zero = zero
  neg_add : ∀ a b : α, neg (add a b) = add (neg a) (neg b)
  two_def : two = add one one
  three_def : three = add two one
  four_def : four = add two two
  half_add_half : add half half = one
  two_mul_half : mul two half = one
  two_ne_zero : two ≠ zero
  div_mul_cancel : ∀ a b : α, b ≠ zero → mul (div a b) b = a
  mul_div_cancel : ∀ a b : α, b ≠ zero → div (mul a b) b = a
  div_mul_div : ∀ a b c d : α, b ≠ zero → d ≠ zero →
    mul (div a b) (div c d) = div (mul a c) (mul b d)
  mul_div_assoc : ∀ a b c : α, mul a (div b c) = div (mul a b) c
  div_div : ∀ a b c : α, b ≠ zero → c ≠ zero →
    div (div a b) c = div a (mul b c)
  div_eq_iff : ∀ a b c : α, b ≠ zero → (div a b = c ↔ a = mul c b)
  mul_ne_zero : ∀ a b : α, a ≠ zero → b ≠ zero → mul a b ≠ zero

section FieldLemmas

variable {α : Type u} [F : ScalarField α]

/-- Cancellation of `(a + b) + (-b) = a` in any `ScalarField`. -/
theorem add_neg_cancel_right (a b : α) : F.add (F.add a b) (F.neg b) = a := by
  rw [F.add_assoc, F.add_neg_cancel, F.add_zero]

/-- Cancellation of `(a + (-b)) + b = a` in any `ScalarField`. -/
theorem neg_add_cancel_right (a b : α) : F.add (F.add a (F.neg b)) b = a := by
  rw [F.add_assoc, F.neg_add_cancel, F.add_zero]

/-- `a - a = 0` in any `ScalarField`. -/
theorem sub_self_eq_zero (a : α) : F.sub a a = F.zero := by
  rw [F.sub_eq_add_neg, F.add_neg_cancel]

/-- `-(a - b) = b - a` in any `ScalarField`. -/
theorem neg_sub_eq (a b : α) : F.neg (F.sub a b) = F.sub b a := by
  rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.neg_add, F.neg_neg, F.add_comm]

/-- Left distributivity over subtraction: `a * (b - c) = a * b - a * c`. -/
theorem mul_sub_left (a b c : α) : F.mul a (F.sub b c) = F.sub (F.mul a b) (F.mul a c) := by
  rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.left_distrib, F.mul_neg]

/-- Right distributivity over subtraction: `(a - b) * c = a * c - b * c`. -/
theorem mul_sub_right (a b c : α) : F.mul (F.sub a b) c = F.sub (F.mul a c) (F.mul b c) := by
  rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.right_distrib, F.neg_mul]

/-- Double half-step sum: `half * x + half * x = x`. -/
theorem half_mul_add_half_mul (x : α) : F.add (F.mul F.half x) (F.mul F.half x) = x := by
  rw [← F.right_distrib, F.half_add_half, F.one_mul]

/-- Commuted `half * two = 1`. -/
theorem half_mul_two : F.mul F.half F.two = F.one := by
  rw [F.mul_comm, F.two_mul_half]

/-- `a / a = 1` for any non-zero scalar `a`. -/
theorem div_self_eq_one (a : α) (ha : a ≠ F.zero) : F.div a a = F.one := by
  have h : a = F.mul F.one a := (F.one_mul a).symm
  exact (F.div_eq_iff a a F.one ha).2 h

/-- Division of a product with matching left factor: `(a * b) / a = b`. -/
theorem mul_div_cancel_left (a b : α) (ha : a ≠ F.zero) : F.div (F.mul a b) a = b := by
  rw [F.mul_comm, F.mul_div_cancel b a ha]

/-- `a - (b + c) = (a - b) - c`. -/
theorem sub_add_eq_sub_sub (a b c : α) : F.sub a (F.add b c) = F.sub (F.sub a b) c := by
  rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.sub_eq_add_neg, F.neg_add, F.add_assoc]

/-- Negating both factors preserves the product: `(-a) * (-b) = a * b`. -/
theorem neg_mul_neg (a b : α) : F.mul (F.neg a) (F.neg b) = F.mul a b := by
  rw [F.neg_mul, F.mul_neg, F.neg_neg]

/-- Rearrangement identity `(a - b) - (c - d) = (a - c) - (b - d)`. -/
theorem sub_sub_swap (a b c d : α) :
    F.sub (F.sub a b) (F.sub c d) = F.sub (F.sub a c) (F.sub b d) := by
  simp only [F.sub_eq_add_neg, F.neg_add, F.neg_neg]
  rw [F.add_assoc, F.add_assoc]
  congr 1
  rw [← F.add_assoc (F.neg b), F.add_comm (F.neg b) (F.neg c), F.add_assoc]

end FieldLemmas

/--
Three-dimensional Cartesian vector `Vec3 α` directly mirroring
`umgebung::scene::Vec3d` in `include/umgebung/scene/UniverseScene.hpp`.
-/
structure Vec3 (α : Type u) where
  x : α
  y : α
  z : α
  deriving DecidableEq, Repr

namespace Vec3

/-- Componentwise extensionality for `Vec3 α`. -/
@[ext]
theorem ext {α : Type u} {u v : Vec3 α} (hx : u.x = v.x) (hy : u.y = v.y) (hz : u.z = v.z) : u = v := by
  cases u; cases v; simp only [mk.injEq]; exact ⟨hx, hy, hz⟩

variable {α : Type u} [F : ScalarField α]

/-- Zero vector `(0, 0, 0)`. -/
def zero : Vec3 α := ⟨F.zero, F.zero, F.zero⟩

/-- Componentwise vector addition `u + v`. -/
def add (u v : Vec3 α) : Vec3 α :=
  ⟨F.add u.x v.x, F.add u.y v.y, F.add u.z v.z⟩

/-- Componentwise vector negation `-u`. -/
def neg (u : Vec3 α) : Vec3 α :=
  ⟨F.neg u.x, F.neg u.y, F.neg u.z⟩

/-- Componentwise vector subtraction `u - v`. -/
def sub (u v : Vec3 α) : Vec3 α :=
  ⟨F.sub u.x v.x, F.sub u.y v.y, F.sub u.z v.z⟩

/-- Scalar multiplication `s * u`. -/
def smul (s : α) (u : Vec3 α) : Vec3 α :=
  ⟨F.mul s u.x, F.mul s u.y, F.mul s u.z⟩

/-- Euclidean dot product `u · v = u_x v_x + u_y v_y + u_z v_z`. -/
def dot (u v : Vec3 α) : α :=
  F.add (F.add (F.mul u.x v.x) (F.mul u.y v.y)) (F.mul u.z v.z)

/-- 3D Cartesian cross product `u × v` (used for angular momentum `r × p`). -/
def cross (u v : Vec3 α) : Vec3 α :=
  ⟨ F.sub (F.mul u.y v.z) (F.mul u.z v.y),
    F.sub (F.mul u.z v.x) (F.mul u.x v.z),
    F.sub (F.mul u.x v.y) (F.mul u.y v.x) ⟩

theorem add_assoc (u v w : Vec3 α) : add (add u v) w = add u (add v w) := by
  apply ext <;> exact F.add_assoc _ _ _

theorem add_comm (u v : Vec3 α) : add u v = add v u := by
  apply ext <;> exact F.add_comm _ _

theorem add_zero (u : Vec3 α) : add u zero = u := by
  apply ext <;> exact F.add_zero _

theorem zero_add (u : Vec3 α) : add zero u = u := by
  apply ext <;> exact F.zero_add _

theorem add_neg_cancel (u : Vec3 α) : add u (neg u) = zero := by
  apply ext <;> exact F.add_neg_cancel _

theorem neg_add_cancel (u : Vec3 α) : add (neg u) u = zero := by
  apply ext <;> exact F.neg_add_cancel _

theorem add_neg_cancel_right (u v : Vec3 α) : add (add u v) (neg v) = u := by
  rw [add_assoc, add_neg_cancel, add_zero]

theorem neg_add_cancel_right (u v : Vec3 α) : add (add u (neg v)) v = u := by
  rw [add_assoc, neg_add_cancel, add_zero]

theorem smul_add (s : α) (u v : Vec3 α) : smul s (add u v) = add (smul s u) (smul s v) := by
  apply ext <;> exact F.left_distrib _ _ _

theorem add_smul (s t : α) (u : Vec3 α) : smul (F.add s t) u = add (smul s u) (smul t u) := by
  apply ext <;> exact F.right_distrib _ _ _

theorem smul_smul (s t : α) (u : Vec3 α) : smul s (smul t u) = smul (F.mul s t) u := by
  apply ext <;> exact (F.mul_assoc _ _ _).symm

theorem one_smul (u : Vec3 α) : smul F.one u = u := by
  apply ext <;> exact F.one_mul _

theorem zero_smul (u : Vec3 α) : smul F.zero u = zero := by
  apply ext <;> exact F.zero_mul _

theorem smul_zero (s : α) : smul s (zero (α := α)) = zero := by
  apply ext <;> exact F.mul_zero _

theorem neg_smul (s : α) (u : Vec3 α) : smul (F.neg s) u = neg (smul s u) := by
  apply ext <;> exact F.neg_mul _ _

theorem smul_neg (s : α) (u : Vec3 α) : smul s (neg u) = neg (smul s u) := by
  apply ext <;> exact F.mul_neg _ _

theorem neg_sub (u v : Vec3 α) : neg (sub u v) = sub v u := by
  apply ext <;> exact neg_sub_eq _ _

/-- Cross product is antisymmetric: `u × v = -(v × u)`. -/
theorem cross_antisymm (u v : Vec3 α) : cross u v = neg (cross v u) := by
  apply ext <;> dsimp [cross, neg]
  · rw [neg_sub_eq, F.mul_comm v.z u.y, F.mul_comm v.y u.z]
  · rw [neg_sub_eq, F.mul_comm v.x u.z, F.mul_comm v.z u.x]
  · rw [neg_sub_eq, F.mul_comm v.y u.x, F.mul_comm v.x u.y]

/-- Self cross product vanishes identically: `u × u = 0`. -/
theorem cross_self (u : Vec3 α) : cross u u = zero := by
  apply ext <;> dsimp [cross, zero]
  · rw [F.mul_comm u.z u.y, sub_self_eq_zero]
  · rw [F.mul_comm u.x u.z, sub_self_eq_zero]
  · rw [F.mul_comm u.y u.x, sub_self_eq_zero]

/-- Right scalar linearity of cross product: `u × (s * v) = s * (u × v)`. -/
theorem cross_smul_right (u v : Vec3 α) (s : α) :
    cross u (smul s v) = smul s (cross u v) := by
  apply ext <;> dsimp [cross, smul] <;> rw [mul_sub_left] <;>
    (congr 1 <;> rw [← F.mul_assoc, F.mul_comm _ s, F.mul_assoc])

/-- Right negation of cross product: `u × (-v) = -(u × v)`. -/
theorem cross_neg_right (u v : Vec3 α) : cross u (neg v) = neg (cross u v) := by
  apply ext <;> dsimp [cross, neg]
  · rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.mul_neg, F.mul_neg, F.neg_add]
  · rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.mul_neg, F.mul_neg, F.neg_add]
  · rw [F.sub_eq_add_neg, F.sub_eq_add_neg, F.mul_neg, F.mul_neg, F.neg_add]

/-- Left distributivity of cross product over subtraction: `(u - v) × w = u × w - v × w`. -/
theorem cross_sub_left (u v w : Vec3 α) :
    cross (sub u v) w = sub (cross u w) (cross v w) := by
  apply ext <;> dsimp [cross, sub] <;> rw [mul_sub_right, mul_sub_right, sub_sub_swap]

/-- Right cross product with a difference: `u × (v - u) = u × v`. -/
theorem cross_sub_self_right (u v : Vec3 α) : cross u (sub v u) = cross u v := by
  have h1 : cross u (sub v u) = neg (cross (sub v u) u) := cross_antisymm u (sub v u)
  rw [h1, cross_sub_left, cross_self]
  have h2 : sub (cross v u) (zero (α := α)) = cross v u := by
    apply ext <;> dsimp [sub, zero] <;> rw [F.sub_eq_add_neg, F.neg_zero, F.add_zero]
  rw [h2, ← cross_antisymm]

end Vec3

end Umgebung.Algebra
