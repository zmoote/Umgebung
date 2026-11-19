import Umgebung.Algebra.ScalarField

/-!
# Umgebung.Numerical.VelocityVerlet

Formal mathematical specification and **sorry-free** verification of the
2nd-order symplectic **Kick-Drift-Kick Velocity Verlet** integrator implemented
in CUDA FP64 in:
- `src/models/mainstream/NewtonianNBodySolver.cu` (`VelocityVerletKickDriftKernel`,
  `VelocityVerletFinalKickKernel`), and
- `src/models/unified/UnifiedFieldSolver.cu` (`UnifiedKickDriftKernel`,
  `UnifiedFinalKickKernel`).

## Proven Theorems
1. `verletStep_time_reversible`:
   Stepping forward by `+dt` and immediately backward by `-dt` recovers the
   exact initial position and velocity `(r₀, v₀)` identically for ANY
   position-dependent acceleration field `a(r)` (corresponding directly to
   `NewtonianNBodyTests.cpp::TestTimeReversibility`, where GPU FP64 achieves
   `1.25e-16` recovery error).
2. `verletTwoBody_conserves_linear_momentum`:
   For ANY pairwise force law satisfying Newton's 3rd Law (`F₁₂ + F₂₁ = 0`,
   covering both Pillar 1 Newtonian gravity and Pillar 2 Love/Phryll pairwise
   coupling), discrete total linear momentum `P = m₁ v₁ + m₂ v₂` is strictly
   invariant across every Velocity Verlet step.
3. `verletNBody_conserves_linear_momentum`:
   Generalization to arbitrary finite `N`-body ensembles (`List` of bodies)
   whenever the net external + internal force sum vanishes (`∑ F_i = 0`).
4. `verletStep_symplectic_jacobian_det_one`:
   Proves that the canonical 1D phase-space Jacobian `J = ∂(q₁, p₁) / ∂(q₀, p₀)`
   of Kick-Drift-Kick Velocity Verlet satisfies `det(J) = 1` identically for all
   timesteps `dt` and arbitrary force gradients `k₀ = F'(q₀)/m`, `k₁ = F'(q₁)/m`,
   establishing exact preservation of the canonical 2-form `dq ∧ dp` (symplecticity).
-/

namespace Umgebung.Numerical

open Umgebung.Algebra

section SingleBodyReversibility

variable {α : Type u} [F : ScalarField α]

/--
Phase-space state `(pos, vel)` of a single particle in 3D Cartesian coordinates.
-/
structure ParticleState (α : Type u) where
  pos : Vec3 α
  vel : Vec3 α
  deriving DecidableEq, Repr

/--
One Kick-Drift-Kick Velocity Verlet step of size `dt` driven by an arbitrary
position-dependent acceleration field `accel : Vec3 α → Vec3 α`:
  1. `v_half = v + (dt / 2) * accel(r)`
  2. `r_next = r + dt * v_half`
  3. `v_next = v_half + (dt / 2) * accel(r_next)`
-/
def verletStep (accel : Vec3 α → Vec3 α) (dt : α) (s : ParticleState α) : ParticleState α :=
  let halfDt := F.mul F.half dt
  let vHalf := Vec3.add s.vel (Vec3.smul halfDt (accel s.pos))
  let rNext := Vec3.add s.pos (Vec3.smul dt vHalf)
  let vNext := Vec3.add vHalf (Vec3.smul halfDt (accel rNext))
  ⟨rNext, vNext⟩

/--
**Theorem (Exact Discrete Time-Reversibility of Velocity Verlet).**
For any position-dependent acceleration field `accel`, any initial state `s`,
and any timestep `dt`, integrating forward by `+dt` and backward by `-dt`
returns `s` identically:

`verletStep accel (-dt) (verletStep accel dt s) = s`.
-/
theorem verletStep_time_reversible
    (accel : Vec3 α → Vec3 α)
    (dt : α)
    (s : ParticleState α) :
    verletStep accel (F.neg dt) (verletStep accel dt s) = s := by
  unfold verletStep
  dsimp only
  -- Let `hdt = half * dt`; then `half * (-dt) = -(half * dt)`.
  have hHalfNeg : F.mul F.half (F.neg dt) = F.neg (F.mul F.half dt) := F.mul_neg F.half dt
  -- Step A: The backward half-kick cancels the forward second half-kick, recovering `vHalf`.
  have hvHalfRec :
      Vec3.add
        (Vec3.add (Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos)))
                  (Vec3.smul (F.mul F.half dt) (accel (Vec3.add s.pos (Vec3.smul dt (Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos))))))))
        (Vec3.smul (F.mul F.half (F.neg dt))
                   (accel (Vec3.add s.pos (Vec3.smul dt (Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos)))))))
      = Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos)) := by
    rw [hHalfNeg, Vec3.neg_smul, Vec3.add_neg_cancel_right]
  -- Step B: The backward drift `(-dt) * vHalf` cancels the forward drift `dt * vHalf`, recovering `s.pos`.
  have hrRec :
      Vec3.add
        (Vec3.add s.pos (Vec3.smul dt (Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos)))))
        (Vec3.smul (F.neg dt) (Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos))))
      = s.pos := by
    rw [Vec3.neg_smul, Vec3.add_neg_cancel_right]
  -- Step C: The final backward half-kick cancels the initial forward half-kick, recovering `s.vel`.
  have hvRec :
      Vec3.add
        (Vec3.add s.vel (Vec3.smul (F.mul F.half dt) (accel s.pos)))
        (Vec3.smul (F.mul F.half (F.neg dt)) (accel s.pos))
      = s.vel := by
    rw [hHalfNeg, Vec3.neg_smul, Vec3.add_neg_cancel_right]
  rw [hvHalfRec, hrRec, hvRec]

end SingleBodyReversibility

section MomentumConservation

variable {α : Type u} [F : ScalarField α]

/--
Discrete momentum update for a body of non-zero mass `m` under Velocity Verlet:
`p_next = m * v_next = m * v + (dt / 2) * (F_0 + F_1)` where `a_k = (1/m) * F_k`.
-/
def bodyMomentum (m : α) (v : Vec3 α) : Vec3 α :=
  Vec3.smul m v

/--
One Velocity Verlet velocity update expressed directly via forces `(f0, f1)`
and inverse mass `invM` (with `m * invM = 1`):
`v_next = v + (half * dt) * (invM * f0) + (half * dt) * (invM * f1)`.
-/
def verletVelUpdate (invM dt : α) (v f0 f1 : Vec3 α) : Vec3 α :=
  let halfDt := F.mul F.half dt
  let vHalf := Vec3.add v (Vec3.smul halfDt (Vec3.smul invM f0))
  Vec3.add vHalf (Vec3.smul halfDt (Vec3.smul invM f1))

/--
Helper lemma: Multiplying `verletVelUpdate` by `m` cancels `m * invM = 1`,
yielding the exact impulse-momentum relation:
`m * v_next = m * v + (half * dt) * f0 + (half * dt) * f1`.
-/
theorem bodyMomentum_verletVelUpdate
    (m invM dt : α)
    (hm : F.mul m invM = F.one)
    (v f0 f1 : Vec3 α) :
    bodyMomentum m (verletVelUpdate invM dt v f0 f1) =
      Vec3.add
        (Vec3.add (bodyMomentum m v) (Vec3.smul (F.mul F.half dt) f0))
        (Vec3.smul (F.mul F.half dt) f1) := by
  unfold bodyMomentum verletVelUpdate
  dsimp only
  rw [Vec3.smul_add, Vec3.smul_add, Vec3.smul_smul, Vec3.smul_smul,
      Vec3.smul_smul, Vec3.smul_smul]
  have h1 : F.mul (F.mul m (F.mul F.half dt)) invM = F.mul F.half dt := by
    rw [F.mul_assoc, F.mul_comm (F.mul F.half dt) invM, ← F.mul_assoc, hm, F.one_mul]
  rw [h1]

/--
**Theorem (Exact Discrete Linear Momentum Conservation for Two-Body Antisymmetric Forces).**
Whenever the net forces at both the start (`t`) and end (`t + dt`) of a
Velocity Verlet step obey Newton's 3rd Law (`f1_0 + f2_0 = 0` and
`f1_1 + f2_1 = 0`), total linear momentum `P = m₁ v₁ + m₂ v₂` is strictly
conserved without any discretization drift.
-/
theorem verletTwoBody_conserves_linear_momentum
    (m1 invM1 m2 invM2 dt : α)
    (hm1 : F.mul m1 invM1 = F.one)
    (hm2 : F.mul m2 invM2 = F.one)
    (v1 v2 f1_0 f2_0 f1_1 f2_1 : Vec3 α)
    (hNewton0 : Vec3.add f1_0 f2_0 = Vec3.zero)
    (hNewton1 : Vec3.add f1_1 f2_1 = Vec3.zero) :
    Vec3.add (bodyMomentum m1 (verletVelUpdate invM1 dt v1 f1_0 f1_1))
             (bodyMomentum m2 (verletVelUpdate invM2 dt v2 f2_0 f2_1)) =
      Vec3.add (bodyMomentum m1 v1) (bodyMomentum m2 v2) := by
  rw [bodyMomentum_verletVelUpdate m1 invM1 dt hm1 v1 f1_0 f1_1,
      bodyMomentum_verletVelUpdate m2 invM2 dt hm2 v2 f2_0 f2_1]
  generalize bodyMomentum m1 v1 = p1
  generalize bodyMomentum m2 v2 = p2
  generalize F.mul F.half dt = hdt
  have hSum0 : Vec3.add (Vec3.smul hdt f1_0) (Vec3.smul hdt f2_0) = Vec3.zero := by
    rw [← Vec3.smul_add, hNewton0, Vec3.smul_zero]
  have hSum1 : Vec3.add (Vec3.smul hdt f1_1) (Vec3.smul hdt f2_1) = Vec3.zero := by
    rw [← Vec3.smul_add, hNewton1, Vec3.smul_zero]
  -- Group `(p1 + p2) + (hdt*f1_0 + hdt*f2_0) + (hdt*f1_1 + hdt*f2_1)`.
  calc
    Vec3.add (Vec3.add (Vec3.add p1 (Vec3.smul hdt f1_0)) (Vec3.smul hdt f1_1))
             (Vec3.add (Vec3.add p2 (Vec3.smul hdt f2_0)) (Vec3.smul hdt f2_1))
      = Vec3.add (Vec3.add (Vec3.add p1 (Vec3.smul hdt f1_0))
                           (Vec3.add p2 (Vec3.smul hdt f2_0)))
                 (Vec3.add (Vec3.smul hdt f1_1) (Vec3.smul hdt f2_1)) := by
          rw [Vec3.add_assoc, ← Vec3.add_assoc (Vec3.smul hdt f1_1),
              Vec3.add_comm (Vec3.smul hdt f1_1) (Vec3.add p2 (Vec3.smul hdt f2_0)),
              Vec3.add_assoc (Vec3.add p2 (Vec3.smul hdt f2_0)),
              ← Vec3.add_assoc]
    _ = Vec3.add (Vec3.add p1 (Vec3.smul hdt f1_0))
                 (Vec3.add p2 (Vec3.smul hdt f2_0)) := by
          rw [hSum1, Vec3.add_zero]
    _ = Vec3.add (Vec3.add p1 p2)
                 (Vec3.add (Vec3.smul hdt f1_0) (Vec3.smul hdt f2_0)) := by
          rw [Vec3.add_assoc, ← Vec3.add_assoc (Vec3.smul hdt f1_0),
              Vec3.add_comm (Vec3.smul hdt f1_0) p2,
              Vec3.add_assoc p2, ← Vec3.add_assoc]
    _ = Vec3.add p1 p2 := by
          rw [hSum0, Vec3.add_zero]

/-- Recursively sums a list of 3D vectors (used for N-body total force & momentum). -/
def sumVec3 : List (Vec3 α) → Vec3 α
  | [] => Vec3.zero
  | v :: vs => Vec3.add v (sumVec3 vs)

/-- Vector sum distributes cleanly over pairwise list addition. -/
theorem sumVec3_zipWith_add :
    ∀ (xs ys : List (Vec3 α)),
      xs.length = ys.length →
      sumVec3 (List.zipWith Vec3.add xs ys) = Vec3.add (sumVec3 xs) (sumVec3 ys)
  | [], [], _ => by
      simp [sumVec3, Vec3.add_zero]
  | x :: xs, y :: ys, hLen => by
      simp only [List.length_cons, Nat.add_right_cancel_iff] at hLen
      simp only [List.zipWith_cons_cons, sumVec3]
      rw [sumVec3_zipWith_add xs ys hLen]
      rw [Vec3.add_assoc, ← Vec3.add_assoc y, Vec3.add_comm y (sumVec3 xs),
          Vec3.add_assoc (sumVec3 xs), ← Vec3.add_assoc]

/-- Scalar multiplication factors out of `sumVec3`. -/
theorem sumVec3_map_smul (s : α) :
    ∀ (xs : List (Vec3 α)),
      sumVec3 (xs.map (Vec3.smul s)) = Vec3.smul s (sumVec3 xs)
  | [] => by simp [sumVec3, Vec3.smul_zero]
  | x :: xs => by
      simp only [List.map_cons, sumVec3]
      rw [sumVec3_map_smul s xs, Vec3.smul_add]

/--
**Theorem (Exact Discrete Linear Momentum Conservation for Arbitrary N-Body Systems).**
For any finite `N`-body system whose momenta update via
`p_next[i] = p[i] + (dt/2) * f0[i] + (dt/2) * f1[i]`, whenever total internal
force vanishes at both endpoints (`sumVec3 f0 = 0` and `sumVec3 f1 = 0`), the
total system momentum `sumVec3 p_next` equals `sumVec3 p` identically.
-/
theorem verletNBody_conserves_linear_momentum
    (dt : α)
    (ps f0s f1s : List (Vec3 α))
    (hLen0 : ps.length = f0s.length)
    (hLen1 : ps.length = f1s.length)
    (hForce0 : sumVec3 f0s = Vec3.zero)
    (hForce1 : sumVec3 f1s = Vec3.zero) :
    let hdt := F.mul F.half dt
    let pHalfs := List.zipWith Vec3.add ps (f0s.map (Vec3.smul hdt))
    let pNexts := List.zipWith Vec3.add pHalfs (f1s.map (Vec3.smul hdt))
    sumVec3 pNexts = sumVec3 ps := by
  intro hdt pHalfs pNexts
  have hLenMap0 : ps.length = (f0s.map (Vec3.smul hdt)).length := by
    rw [List.length_map]; exact hLen0
  have hLenHalf : pHalfs.length = (f1s.map (Vec3.smul hdt)).length := by
    simp only [pHalfs, List.length_zipWith, List.length_map]
    rw [← hLen0, Nat.min_self, hLen1]
  have hStep1 : sumVec3 pNexts =
      Vec3.add (sumVec3 pHalfs) (sumVec3 (f1s.map (Vec3.smul hdt))) :=
    sumVec3_zipWith_add pHalfs (f1s.map (Vec3.smul hdt)) hLenHalf
  have hStep0 : sumVec3 pHalfs =
      Vec3.add (sumVec3 ps) (sumVec3 (f0s.map (Vec3.smul hdt))) :=
    sumVec3_zipWith_add ps (f0s.map (Vec3.smul hdt)) hLenMap0
  rw [hStep1, hStep0, sumVec3_map_smul, sumVec3_map_smul,
      hForce0, hForce1, Vec3.smul_zero, Vec3.add_zero, Vec3.add_zero]

end MomentumConservation

section Symplecticity

variable {α : Type u} [F : ScalarField α]

/--
A `2 × 2` canonical phase-space Jacobian matrix `[[j11, j12], [j21, j22]]`
representing the differential `d(q₁, p₁) / d(q₀, p₀)` of a 1D canonical map.
-/
structure Mat2 (α : Type u) where
  a11 : α
  a12 : α
  a21 : α
  a22 : α
  deriving DecidableEq, Repr

/-- Matrix multiplication for `2 × 2` phase-space Jacobians. -/
def Mat2.mul (A B : Mat2 α) : Mat2 α :=
  ⟨ F.add (F.mul A.a11 B.a11) (F.mul A.a12 B.a21),
    F.add (F.mul A.a11 B.a12) (F.mul A.a12 B.a22),
    F.add (F.mul A.a21 B.a11) (F.mul A.a22 B.a21),
    F.add (F.mul A.a21 B.a12) (F.mul A.a22 B.a22) ⟩

/-- Determinant `a11 * a22 - a12 * a21` of a `2 × 2` Jacobian (equals `1` iff symplectic in 2D phase space). -/
def Mat2.det (A : Mat2 α) : α :=
  F.sub (F.mul A.a11 A.a22) (F.mul A.a12 A.a21)

/--
Jacobian of a single Kick stage `q' = q, p' = p + h * F(q)` with local force
derivative `k = F'(q)`: `[[1, 0], [h * k, 1]]`.
-/
def kickJacobian (h k : α) : Mat2 α :=
  ⟨F.one, F.zero, F.mul h k, F.one⟩

/--
Jacobian of a single Drift stage `q' = q + dt * (p / m), p' = p` with inverse
mass `invM`: `[[1, dt * invM], [0, 1]]`.
-/
def driftJacobian (dt invM : α) : Mat2 α :=
  ⟨F.one, F.mul dt invM, F.zero, F.one⟩

/-- Determinant of a Kick shear matrix is identically `1`. -/
theorem kickJacobian_det (h k : α) : (kickJacobian h k).det = F.one := by
  unfold Mat2.det kickJacobian
  dsimp only
  rw [F.one_mul, F.zero_mul, F.sub_eq_add_neg, F.neg_zero, F.add_zero]

/-- Determinant of a Drift shear matrix is identically `1`. -/
theorem driftJacobian_det (dt invM : α) : (driftJacobian dt invM).det = F.one := by
  unfold Mat2.det driftJacobian
  dsimp only
  rw [F.one_mul, F.mul_zero, F.sub_eq_add_neg, F.neg_zero, F.add_zero]

/--
Multiplicativity of `2 × 2` determinants (`det(A * B) = det(A) * det(B)`)
proven directly from `ScalarField` axioms without external libraries.
-/
theorem Mat2.det_mul (A B : Mat2 α) :
    (Mat2.mul A B).det = F.mul A.det B.det := by
  unfold Mat2.det Mat2.mul
  dsimp only
  -- Expand both sides and cancel the four matching cross terms `(a11*a12*b11*b22)` etc.
  rw [F.left_distrib, F.right_distrib, F.right_distrib,
      F.left_distrib, F.right_distrib, F.right_distrib,
      mul_sub_left, mul_sub_right, mul_sub_right]
  -- Normalize every 4-term monomial into canonical `a_i * a_j * b_k * b_l` ordering.
  have t1 : F.mul (F.mul A.a11 B.a11) (F.mul A.a21 B.a12) =
            F.mul (F.mul A.a11 B.a12) (F.mul A.a21 B.a11) := by
    rw [F.mul_assoc, F.mul_assoc, ← F.mul_assoc B.a11, F.mul_comm B.a11 A.a21,
        F.mul_assoc A.a21, F.mul_comm B.a11 B.a12, ← F.mul_assoc A.a21,
        F.mul_comm A.a21 B.a12, F.mul_assoc B.a12]
  have t2 : F.mul (F.mul A.a12 B.a21) (F.mul A.a22 B.a22) =
            F.mul (F.mul A.a12 B.a22) (F.mul A.a22 B.a21) := by
    rw [F.mul_assoc, F.mul_assoc, ← F.mul_assoc B.a21, F.mul_comm B.a21 A.a22,
        F.mul_assoc A.a22, F.mul_comm B.a21 B.a22, ← F.mul_assoc A.a22,
        F.mul_comm A.a22 B.a22, F.mul_assoc B.a22]
  have m1 : F.mul (F.mul A.a11 B.a11) (F.mul A.a22 B.a22) =
            F.mul (F.mul A.a11 A.a22) (F.mul B.a11 B.a22) := by
    rw [F.mul_assoc, F.mul_assoc, ← F.mul_assoc B.a11, F.mul_comm B.a11 A.a22, F.mul_assoc]
  have m2 : F.mul (F.mul A.a12 B.a21) (F.mul A.a21 B.a12) =
            F.mul (F.mul A.a12 A.a21) (F.mul B.a12 B.a21) := by
    rw [F.mul_assoc, F.mul_assoc, ← F.mul_assoc B.a21, F.mul_comm B.a21 A.a21,
        F.mul_assoc A.a21, F.mul_comm B.a21 B.a12]
  have m3 : F.mul (F.mul A.a11 B.a12) (F.mul A.a22 B.a21) =
            F.mul (F.mul A.a11 A.a22) (F.mul B.a12 B.a21) := by
    rw [F.mul_assoc, F.mul_assoc, ← F.mul_assoc B.a12, F.mul_comm B.a12 A.a22, F.mul_assoc]
  have m4 : F.mul (F.mul A.a12 B.a22) (F.mul A.a21 B.a11) =
            F.mul (F.mul A.a12 A.a21) (F.mul B.a11 B.a22) := by
    rw [F.mul_assoc, F.mul_assoc, ← F.mul_assoc B.a22, F.mul_comm B.a22 A.a21,
        F.mul_assoc A.a21, F.mul_comm B.a22 B.a11]
  rw [t1, t2, m1, m2, m3, m4]
  generalize F.mul (F.mul A.a11 B.a12) (F.mul A.a21 B.a11) = x
  generalize F.mul (F.mul A.a12 B.a22) (F.mul A.a22 B.a21) = y
  generalize F.mul (F.mul A.a11 A.a22) (F.mul B.a11 B.a22) = p
  generalize F.mul (F.mul A.a12 A.a21) (F.mul B.a12 B.a21) = q
  generalize F.mul (F.mul A.a11 A.a22) (F.mul B.a12 B.a21) = r
  generalize F.mul (F.mul A.a12 A.a21) (F.mul B.a11 B.a22) = s
  -- Goal: `((x + q) + (p + y)) - ((x + s) + (r + y)) = (p - s) - (r - q)`.
  simp only [F.sub_eq_add_neg, F.neg_add, F.neg_neg]
  -- Rearrange `(x + q) + (p + y) = (x + y) + (p + q)`.
  have hL : F.add (F.add x q) (F.add p y) = F.add (F.add x y) (F.add p q) := by
    rw [F.add_comm p y, F.add_assoc, ← F.add_assoc q y p, F.add_comm q y,
        F.add_assoc y q p, ← F.add_assoc, F.add_comm q p]
  -- Rearrange `(-x + -s) + (-r + -y) = (-x + -y) + (-s + -r)`.
  have hR : F.add (F.add (F.neg x) (F.neg s)) (F.add (F.neg r) (F.neg y)) =
            F.add (F.add (F.neg x) (F.neg y)) (F.add (F.neg s) (F.neg r)) := by
    rw [F.add_comm (F.neg r) (F.neg y), F.add_assoc, ← F.add_assoc (F.neg s) (F.neg y),
        F.add_comm (F.neg s) (F.neg y), F.add_assoc (F.neg y), ← F.add_assoc]
  rw [hL, hR]
  have hXY : F.add (F.add x y) (F.add (F.neg x) (F.neg y)) = F.zero := by
    rw [← F.neg_add, F.add_neg_cancel]
  rw [F.add_assoc, ← F.add_assoc (F.add p q), F.add_comm (F.add p q) (F.add (F.neg x) (F.neg y)),
      F.add_assoc (F.add (F.neg x) (F.neg y)), ← F.add_assoc (F.add x y), hXY, F.zero_add]
  -- Reorder `(p + q) + (-s + -r) = (p + -s) + (-r + q)`.
  rw [F.add_assoc, ← F.add_assoc q, F.add_comm q (F.neg s), F.add_assoc (F.neg s),
      F.add_comm q (F.neg r), ← F.add_assoc]

/--
Full canonical phase-space Jacobian of one Kick-Drift-Kick Velocity Verlet step:
`J_VV = J_kick(dt/2, k₁) · J_drift(dt, invM) · J_kick(dt/2, k₀)`.
-/
def verletJacobian (dt invM k0 k1 : α) : Mat2 α :=
  let hdt := F.mul F.half dt
  Mat2.mul (kickJacobian hdt k1) (Mat2.mul (driftJacobian dt invM) (kickJacobian hdt k0))

/--
**Theorem (Exact Symplecticity of Kick-Drift-Kick Velocity Verlet).**
For any timestep `dt`, inverse mass `invM`, and arbitrary local potential
curvatures `k0 = -V''(q₀)` and `k1 = -V''(q₁)`, the Velocity Verlet phase-space
Jacobian has determinant strictly equal to `1`:

`(verletJacobian dt invM k0 k1).det = 1`.

This proves exact preservation of canonical 2D phase-space area `dq ∧ dp`
(Liouville's Theorem / symplecticity), explaining why Velocity Verlet bounds
total energy error without secular drift over 20,000+ orbits on the GPU.
-/
theorem verletStep_symplectic_jacobian_det_one (dt invM k0 k1 : α) :
    (verletJacobian dt invM k0 k1).det = F.one := by
  unfold verletJacobian
  dsimp only
  rw [Mat2.det_mul, Mat2.det_mul, kickJacobian_det, driftJacobian_det,
      kickJacobian_det, F.one_mul, F.one_mul]

end Symplecticity

end Umgebung.Numerical
