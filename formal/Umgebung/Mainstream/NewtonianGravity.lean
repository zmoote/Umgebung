import Umgebung.Algebra.ScalarField

/-!
# Umgebung.Mainstream.NewtonianGravity

Formal specification and **sorry-free** conservation proofs for Pillar 1
(`models/mainstream/NewtonianNBodySolver.cu`):
1. `newtonianPairForce_antisymm`:
   Pairwise Newtonian gravity obeys Newton's 3rd Law (`F_ij = -F_ji`), which
   directly feeds `Umgebung.Numerical.verletTwoBody_conserves_linear_momentum`.
2. `centralForce_torque_cancels`:
   Any radial central force `F_ij = s * (r_j - r_i)` satisfies
   `r_i × F_ij + r_j × F_ji = 0`, establishing exact angular momentum
   conservation `dL/dt = 0` for pairwise gravitational systems.
3. `kepler_circular_virial_energy`:
   For a circular Kepler orbit satisfying centripetal balance `v² = G M / r`
   (`scenes/kepler_two_body.umgebung`), proves the Virial Theorem `T = -(1/2) U`
   and total mechanical energy `E = T + U = -(1/2) (G M m / r)`.
-/

namespace Umgebung.Mainstream

open Umgebung.Algebra

section PairwiseCentralForces

variable {α : Type u} [F : ScalarField α]

/--
General radial central pairwise force exerted on body `i` (at `ri`) by body `j`
(at `rj`) with symmetric scalar radial factor `radialCoeff` (for Newtonian
gravity, `radialCoeff = G * m_i * m_j / (|r_j - r_i|² + ε²)^(3/2)`):
`F_ij = radialCoeff * (r_j - r_i)`.
-/
def centralPairForce (radialCoeff : α) (ri rj : Vec3 α) : Vec3 α :=
  Vec3.smul radialCoeff (Vec3.sub rj ri)

/--
**Theorem (Newton's Third Law — Action-Reaction Antisymmetry).**
Whenever the radial scalar factor is symmetric between `i` and `j`, the
pairwise gravitational force satisfies `F_ij = -F_ji` and `F_ij + F_ji = 0`.
-/
theorem newtonianPairForce_antisymm (radialCoeff : α) (ri rj : Vec3 α) :
    centralPairForce radialCoeff ri rj = Vec3.neg (centralPairForce radialCoeff rj ri) := by
  unfold centralPairForce
  rw [← Vec3.smul_neg, Vec3.neg_sub]

/-- Pairwise force sum vanishes identically: `F_ij + F_ji = 0`. -/
theorem newtonianPairForce_sum_zero (radialCoeff : α) (ri rj : Vec3 α) :
    Vec3.add (centralPairForce radialCoeff ri rj) (centralPairForce radialCoeff rj ri) = Vec3.zero := by
  rw [newtonianPairForce_antisymm radialCoeff ri rj, Vec3.neg_add_cancel]

/--
**Theorem (Exact Pairwise Torque Cancellation / Angular Momentum Conservation).**
For any radial central force `F_ij = radialCoeff * (r_j - r_i)` (including
Newtonian gravity in `NewtonianNBodySolver.cu`), the sum of pairwise torques
about the origin vanishes identically:

`r_i × F_ij + r_j × F_ji = 0`.
-/
theorem centralForce_torque_cancels (radialCoeff : α) (ri rj : Vec3 α) :
    Vec3.add (Vec3.cross ri (centralPairForce radialCoeff ri rj))
             (Vec3.cross rj (centralPairForce radialCoeff rj ri)) = Vec3.zero := by
  unfold centralPairForce
  rw [Vec3.cross_smul_right, Vec3.cross_smul_right,
      Vec3.cross_sub_self_right ri rj, Vec3.cross_sub_self_right rj ri,
      Vec3.cross_antisymm rj ri, Vec3.smul_neg, Vec3.add_neg_cancel]

end PairwiseCentralForces

section KeplerCircularBenchmark

variable {α : Type u} [F : ScalarField α]

/--
Kinetic energy `T = (1/2) * m * v²` of an orbiting body.
-/
def kineticEnergy (m vSq : α) : α :=
  F.mul F.half (F.mul m vSq)

/--
Gravitational potential energy `U = -(G * M * m / r)` of a two-body pair at separation `r`.
-/
def gravitationalPotentialEnergy (G M m r : α) : α :=
  F.neg (F.div (F.mul (F.mul G M) m) r)

/--
**Theorem (Kepler Circular Orbit Virial Identity & Exact Total Energy).**
For a circular orbit in `scenes/kepler_two_body.umgebung` obeying the
centripetal balance `v² = (G * M) / r`:
1. `T = -(1/2) * U` (Virial Theorem), and
2. `E = T + U = -(1/2) * (G * M * m / r)`.
-/
theorem kepler_circular_virial_energy
    (G M m r vSq : α)
    (hCirc : vSq = F.div (F.mul G M) r) :
    F.add (kineticEnergy m vSq) (gravitationalPotentialEnergy G M m r) =
      F.neg (F.mul F.half (F.div (F.mul (F.mul G M) m) r)) := by
  unfold kineticEnergy gravitationalPotentialEnergy
  rw [hCirc]
  have hT : F.mul F.half (F.mul m (F.div (F.mul G M) r)) =
            F.mul F.half (F.div (F.mul (F.mul G M) m) r) := by
    congr 1
    rw [F.mul_div_assoc, F.mul_comm m (F.mul G M)]
  rw [hT]
  generalize F.div (F.mul (F.mul G M) m) r = uMag
  -- Write `-uMag = -(half * uMag + half * uMag)` and cancel `half * uMag`.
  have hSplit : F.neg uMag = F.add (F.neg (F.mul F.half uMag)) (F.neg (F.mul F.half uMag)) := by
    rw [← F.neg_add, half_mul_add_half_mul]
  rw [hSplit, ← F.add_assoc, F.add_neg_cancel, F.zero_add]

end KeplerCircularBenchmark

end Umgebung.Mainstream
