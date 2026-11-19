import Umgebung.Algebra.ScalarField

/-!
# Umgebung.Unified.FlowerOfLife

Formal specification and **sorry-free** algebraic proofs for Elena Danaan's
**Flower of Life Multiverse** lattice (`src/scene/FlowerOfLifeMultiverse.cpp`)
and the newly modularized standalone Thinker invariants (`models/thinkers/`):

1. **Flower of Life / Seed of Life Equilateral Geometry:**
   - `flowerOfLife_ring1_0_distSq_origin`: Ring-1 node 0 lies at exact distance `R` from the center Seed.
   - `flowerOfLife_ring1_1_distSq_origin`: Ring-1 node 1 (`60°`) lies at exact distance `R` from the center Seed.
   - `flowerOfLife_vesicaPiscis_equilateral`: Adjacent Ring-1 nodes `0` and `1` sit at exact distance `R` from each other, forming an exact equilateral Vesica Piscis triangle of side `R`.
   - `flowerOfLife_ring1_1_2_equilateral`: Ring-1 nodes `1` (`60°`) and `2` (`120°`) sit at exact distance `R` from each other.
2. **Haramein-Rauscher Spacetime Torque (`HarameinModel.hpp`):**
   - `harameinCoriolis_zero_power`: The Coriolis acceleration `a_Cor = -2 Ω × v` is strictly orthogonal to velocity (`v · a_Cor = 0`), performing identically zero spurious mechanical work.
3. **Salvatore Pais HEEMFG Inertial Mass Reduction (`PaisModel.hpp`):**
   - `paisInertialFactor_inactive`: When `κ = 0`, `m_eff / m_0 = 1`.
   - `paisInertialFactor_mul_amplification`: `(m_eff / m_0) * (m_0 / m_eff) = 1`.
4. **Vogel / Willis Phryll Crystal Saturation (`VogelWillisModel.hpp`):**
   - `vogelSaturationRatio_at_threshold`: At `E_cum = E_sat ≠ 0`, saturation ratio equals exactly `1` (the audible "singing" boundary).
-/

namespace Umgebung.Unified

open Umgebung.Algebra

section FlowerOfLifeGeometry

variable {α : Type u} [F : ScalarField α]

/-- Squared Euclidean distance `‖u - v‖² = (u - v) · (u - v)` between two 3D lattice centers. -/
def sqDist (u v : Vec3 α) : α :=
  let d := Vec3.sub u v
  Vec3.dot d d

/-- Center Seed Bubble Universe at the Flower of Life origin `(0, 0, 0)`. -/
def seedOrigin : Vec3 α := Vec3.zero

/-- Ring-1 node 0 (`0°`): `(R, 0, 0)`. -/
def ring1Node0 (R : α) : Vec3 α := ⟨R, F.zero, F.zero⟩

/-- Ring-1 node 1 (`60°`): `((1/2)*R, s*R, 0)` where `s = √3 / 2` satisfies `s² = 3/4`. -/
def ring1Node1 (s R : α) : Vec3 α := ⟨F.mul F.half R, F.mul s R, F.zero⟩

/-- Ring-1 node 2 (`120°`): `(-(1/2)*R, s*R, 0)` where `s = √3 / 2` satisfies `s² = 3/4`. -/
def ring1Node2 (s R : α) : Vec3 α := ⟨F.neg (F.mul F.half R), F.mul s R, F.zero⟩

/-- Algebraic helper: `half * half + 3 / 4 = 1` in any `ScalarField` with `4 ≠ 0`. -/
theorem half_sq_add_three_fourths (h4 : F.four ≠ F.zero) :
    F.add (F.mul F.half F.half) (F.div F.three F.four) = F.one := by
  have hFourEq : F.four = F.mul F.two F.two := by
    rw [F.four_def]
    calc
      F.add F.two F.two = F.add (F.mul F.two F.one) (F.mul F.two F.one) := by rw [F.mul_one]
      _ = F.mul F.two (F.add F.one F.one) := (F.left_distrib _ _ _).symm
      _ = F.mul F.two F.two := by rw [← F.two_def]
  have hHalfSq : F.mul F.half F.half = F.div F.one F.four := by
    have hProd : F.mul (F.mul F.half F.half) F.four = F.one := by
      rw [hFourEq, ← F.mul_assoc, F.mul_assoc F.half F.half F.two, half_mul_two, F.mul_one, half_mul_two]
    exact ((F.div_eq_iff F.one F.four (F.mul F.half F.half) h4).2 hProd.symm).symm
  rw [hHalfSq]
  -- Now prove `1/4 + 3/4 = 1` via `mul_right` by `4 ≠ 0`.
  have hAddDiv :
      F.mul (F.add (F.div F.one F.four) (F.div F.three F.four)) F.four = F.four := by
    rw [F.right_distrib, F.div_mul_cancel F.one F.four h4, F.div_mul_cancel F.three F.four h4]
    rw [F.add_comm F.one F.three, F.three_def, F.add_assoc, ← F.two_def, ← F.four_def]
  have hDivEq :
      F.add (F.div F.one F.four) (F.div F.three F.four) = F.div F.four F.four := by
    exact ((F.div_eq_iff F.four F.four _ h4).2 hAddDiv.symm).symm
  rw [hDivEq, div_self_eq_one F.four h4]

/--
**Theorem (Ring-1 Node 0 Distance from Center Seed).**
The center-to-node-0 squared distance equals `R²`.
-/
theorem flowerOfLife_ring1_0_distSq_origin (R : α) :
    sqDist (ring1Node0 R) seedOrigin = F.mul R R := by
  unfold sqDist ring1Node0 seedOrigin Vec3.sub Vec3.dot Vec3.zero
  simp only [F.sub_eq_add_neg, F.neg_zero, F.add_zero, F.zero_mul]

/--
**Theorem (Ring-1 Node 1 Distance from Center Seed).**
Whenever `s² = 3/4` (`s = √3 / 2`) and `4 ≠ 0`, Ring-1 node 1 (`60°`) lies at
exact squared distance `R²` from the center Seed Bubble Universe.
-/
theorem flowerOfLife_ring1_1_distSq_origin (s R : α)
    (h4 : F.four ≠ F.zero) (hs : F.mul s s = F.div F.three F.four) :
    sqDist (ring1Node1 s R) seedOrigin = F.mul R R := by
  unfold sqDist ring1Node1 seedOrigin Vec3.sub Vec3.dot Vec3.zero
  simp only [F.sub_eq_add_neg, F.neg_zero, F.add_zero, F.zero_mul]
  have hx : F.mul (F.mul F.half R) (F.mul F.half R) = F.mul (F.mul F.half F.half) (F.mul R R) := by
    rw [F.mul_assoc, ← F.mul_assoc R F.half R, F.mul_comm R F.half, F.mul_assoc F.half R R, ← F.mul_assoc]
  have hy : F.mul (F.mul s R) (F.mul s R) = F.mul (F.mul s s) (F.mul R R) := by
    rw [F.mul_assoc, ← F.mul_assoc R s R, F.mul_comm R s, F.mul_assoc s R R, ← F.mul_assoc]
  rw [hx, hy, hs, ← F.right_distrib, half_sq_add_three_fourths h4, F.one_mul]

/--
**Theorem (Flower of Life Vesica Piscis Equilateral Invariant — Nodes 0 & 1).**
The squared distance between adjacent Ring-1 Bubble Universes `ring1Node0` (`0°`)
and `ring1Node1` (`60°`) equals `R²`, proving that `{origin, ring1Node0, ring1Node1}`
forms an exact equilateral triangle of side length `R` (`FlowerOfLifeMultiverse.cpp`).
-/
theorem flowerOfLife_vesicaPiscis_equilateral (s R : α)
    (h4 : F.four ≠ F.zero) (hs : F.mul s s = F.div F.three F.four) :
    sqDist (ring1Node0 R) (ring1Node1 s R) = F.mul R R := by
  unfold sqDist ring1Node0 ring1Node1 Vec3.sub Vec3.dot
  simp only [F.sub_eq_add_neg, F.neg_zero, F.add_zero, F.zero_add, F.zero_mul]
  -- Simplify `R + -(half * R) = half * R`.
  have hRx : F.add R (F.neg (F.mul F.half R)) = F.mul F.half R := by
    calc
      F.add R (F.neg (F.mul F.half R))
          = F.add (F.add (F.mul F.half R) (F.mul F.half R)) (F.neg (F.mul F.half R)) := by
              rw [half_mul_add_half_mul]
      _ = F.mul F.half R := add_neg_cancel_right (F.mul F.half R) (F.mul F.half R)
  rw [hRx, neg_mul_neg]
  have hx : F.mul (F.mul F.half R) (F.mul F.half R) = F.mul (F.mul F.half F.half) (F.mul R R) := by
    rw [F.mul_assoc, ← F.mul_assoc R F.half R, F.mul_comm R F.half, F.mul_assoc F.half R R, ← F.mul_assoc]
  have hy : F.mul (F.mul s R) (F.mul s R) = F.mul (F.mul s s) (F.mul R R) := by
    rw [F.mul_assoc, ← F.mul_assoc R s R, F.mul_comm R s, F.mul_assoc s R R, ← F.mul_assoc]
  rw [hx, hy, hs, ← F.right_distrib, half_sq_add_three_fourths h4, F.one_mul]

/--
**Theorem (Flower of Life Adjacent Ring-1 Spacing — Nodes 1 & 2).**
The squared distance between Ring-1 Bubble Universes at `60°` and `120°` equals `R²`.
-/
theorem flowerOfLife_ring1_1_2_equilateral (s R : α) :
    sqDist (ring1Node1 s R) (ring1Node2 s R) = F.mul R R := by
  unfold sqDist ring1Node1 ring1Node2 Vec3.sub Vec3.dot
  simp only [F.sub_eq_add_neg, F.neg_neg, F.neg_zero, F.add_zero, F.add_neg_cancel, F.zero_mul]
  rw [half_mul_add_half_mul]

end FlowerOfLifeGeometry

section StandaloneThinkerInvariants

variable {α : Type u} [F : ScalarField α]

/--
Haramein-Rauscher spacetime-torque Coriolis acceleration
`a_Cor = -2 Ω × v = (2 Ω_z v_y, -2 Ω_z v_x, 0)` (`HarameinModel.hpp`).
-/
def harameinCoriolisAccel (omegaZ : α) (v : Vec3 α) : Vec3 α :=
  ⟨F.mul (F.mul F.two omegaZ) v.y, F.neg (F.mul (F.mul F.two omegaZ) v.x), F.zero⟩

/--
**Theorem (Zero Mechanical Power of Haramein-Rauscher Spacetime Torque).**
The Coriolis spacetime-torque acceleration is strictly perpendicular to velocity:
`v · a_Cor = 0`,
confirming that Strang-split Coriolis rotation in `HarameinSolver.cu` imparts
spin/precession without injecting or draining spurious kinetic energy.
-/
theorem harameinCoriolis_zero_power (omegaZ : α) (v : Vec3 α) :
    Vec3.dot v (harameinCoriolisAccel omegaZ v) = F.zero := by
  unfold harameinCoriolisAccel Vec3.dot
  rw [F.mul_zero, F.add_zero, F.mul_neg]
  have hTerm :
      F.mul v.y (F.mul (F.mul F.two omegaZ) v.x) =
      F.mul v.x (F.mul (F.mul F.two omegaZ) v.y) := by
    rw [← F.mul_assoc, F.mul_comm v.y (F.mul F.two omegaZ), F.mul_assoc,
        F.mul_comm v.y v.x, ← F.mul_assoc, F.mul_comm (F.mul F.two omegaZ) v.x, F.mul_assoc]
  rw [hTerm, F.add_neg_cancel]

/-- Pais HEEMFG inertial mass factor `m_eff / m_0 = 1 / (1 + κ S_max)` (`PaisModel.hpp`). -/
def paisInertialFactor (sMax kappa : α) : α :=
  F.div F.one (F.add F.one (F.mul kappa sMax))

/-- Pais HEEMFG acceleration amplification `m_0 / m_eff = 1 + κ S_max` (`PaisModel.hpp`). -/
def paisAccelAmplification (sMax kappa : α) : α :=
  F.add F.one (F.mul kappa sMax)

/-- **Theorem (Pais Inactive Cavity Limit):** At `κ = 0`, `m_eff / m_0 = 1`. -/
theorem paisInertialFactor_inactive (sMax : α) (h1 : F.one ≠ F.zero) :
    paisInertialFactor sMax F.zero = F.one := by
  unfold paisInertialFactor
  rw [F.zero_mul, F.add_zero, div_self_eq_one F.one h1]

/-- **Theorem (Pais Reciprocal Duality):** `(m_eff / m_0) * (m_0 / m_eff) = 1`. -/
theorem paisInertialFactor_mul_amplification (sMax kappa : α)
    (hDen : F.add F.one (F.mul kappa sMax) ≠ F.zero) :
    F.mul (paisInertialFactor sMax kappa) (paisAccelAmplification sMax kappa) = F.one := by
  unfold paisInertialFactor paisAccelAmplification
  exact F.div_mul_cancel F.one (F.add F.one (F.mul kappa sMax)) hDen

/-- Vogel/Willis Phryll crystal saturation ratio `E_cum / E_sat` (`VogelWillisModel.hpp`). -/
def vogelSaturationRatio (cumulativeEnergy saturationThreshold : α) : α :=
  F.div cumulativeEnergy saturationThreshold

/-- **Theorem (Vogel/Willis Crystal Singing Boundary):** At `E_cum = E_sat ≠ 0`, ratio = `1`. -/
theorem vogelSaturationRatio_at_threshold (E_sat : α) (hSat : E_sat ≠ F.zero) :
    vogelSaturationRatio E_sat E_sat = F.one :=
  div_self_eq_one E_sat hSat

end StandaloneThinkerInvariants

end Umgebung.Unified
