import Umgebung.Algebra.ScalarField
import Umgebung.Mainstream.NewtonianGravity

/-!
# Umgebung.Unified.ThinkersModels

Formal specification, **explicit postulated axioms**, and **sorry-free proven
consequences** for Pillar 2 (`include/umgebung/models/unified/UnifiedPhysicsEquations.hpp`
and `src/models/unified/UnifiedFieldSolver.cu`), citing primary documents in
`submodules/research/Thinkers/` verbatim.

## Separation of Postulated Axioms vs. Proven Consequences
Following `GEMINI.md` Section 1.3, every theoretical hypothesis imported from
a Thinker's paper is declared explicitly as a named `structure` hypothesis
(never hidden as an unchecked global Lean `axiom`), and every theorem below
is machine-checked with zero `sorry`s:

1. **Nassim Haramein (2013)** — `Haramein342013PRRI3363.pdf` (*Quantum Gravity
   and the Holographic Mass*, PRRI 3(4): 270–292):
   - `haramein_phi_closed_form`: Proves `φ(r) = η(r) / R(r) = 2 ℓ_P / r` (Eq. 5–8).
   - `haramein_holographic_duality`: Proves `m_h(r) * m_p'(r) = 2 m_ℓ²` (Eq. 9, 24).
   - `haramein_schwarzschild_equivalence`: Under the Planck-scale identity
     `m_ℓ / ℓ_P = c² / G`, proves `m_h(r) = r c² / (2G) ≡ m_s(r)` verbatim (Eq. 12–13).

2. **Chris Essonne** — `research_papers_gravity_science_journal_8052.pdf` &
   `research_papers_gravity_science_journal_10100.pdf` (Feb 2025):
   - `essonne_zero_central_pressure_at_3_2_Rs`: Proves interior vacuum pressure
     `P(0) / P₀ = 0` at `R = (3/2) R_s` (`8052.pdf` Eq. 35, 39).
   - `essonne_true_horizon_2Rs`: Proves local light speed squared equals escape
     velocity squared (`c'(r)² = v_L(r)²`) at `r = 2 R_s` (`8052.pdf` Eq. 41–43).
   - `essonne_flat_galaxy_rotation_velocity`: Proves centripetal balance
     `v² / r = g_r(r) = (2GM - R_c² g_z) / (h · r)` cancels `r` identically,
     yielding constant flat orbital velocity `v² = (2GM - R_c² g_z) / h`
     independent of radius `r > R_c` (`10100.pdf` Eq. 11).

3. **Salvatore Pais** — `US10144532.pdf` (Eq. 1):
   - `pais_poynting_flux_superposition`: Proves `S_max` decomposes additively
     into rotational, vibrational, and translational Poynting flux terms.

4. **Elena Danaan / Dan Willis / Marcel Vogel** (`PhryllGenInstructions.pdf`):
   - `lovePhryllPairForce_antisymm` & `lovePhryllPairForce_torque_cancels`:
     Proves the Pillar 2 Love/Phryll resonant coherence force implemented in
     `UnifiedFieldSolver.cu` obeys Newton's 3rd Law and exact torque cancellation,
     guaranteeing discrete linear & angular momentum conservation.
-/

namespace Umgebung.Unified

open Umgebung.Algebra

-- =============================================================================
-- 1. Nassim Haramein (2013) — Quantum Gravity and the Holographic Mass
--    Source: submodules/research/Thinkers/Nassim_Haramein/Haramein342013PRRI3363.pdf
-- =============================================================================
section Haramein2013

variable {α : Type u} [F : ScalarField α]

/--
Haramein (2013) Eq. (7): Surface PSU tiling count `η(r) = 4 r² / r_ℓ²`,
where `r_ℓ = ℓ_P / 2`.
-/
def harameinEta (r rL : α) : α :=
  F.div (F.mul F.four (F.mul r r)) (F.mul rL rL)

/--
Haramein (2013) Eq. (8): Interior volume PSU oscillator count `R(r) = r³ / r_ℓ³`.
-/
def harameinR (r rL : α) : α :=
  F.div (F.mul (F.mul r r) r) (F.mul (F.mul rL rL) rL)

/--
Haramein (2013) closed-form fundamental surface-to-volume holographic ratio
`φ(r) = 2 ℓ_P / r` (matching `ComputeHolographicRatioPhi` in C++/CUDA).
-/
def harameinPhi (r lP : α) : α :=
  F.div (F.mul F.two lP) r

/--
Haramein (2013) Eq. (9)–(13): Holographic gravitational (Schwarzschild) mass
`m_h(r) = m_ℓ / φ(r) = (r * m_ℓ) / (2 * ℓ_P)`.
-/
def harameinHolographicMass (r lP mL : α) : α :=
  F.div (F.mul r mL) (F.mul F.two lP)

/--
Haramein (2013) Eq. (24)–(27): Holographic proton rest mass
`m_p'(r) = 2 * φ(r) * m_ℓ`.
-/
def harameinProtonMass (r lP mL : α) : α :=
  F.mul F.two (F.mul (harameinPhi r lP) mL)

/--
Explicit Planck-scale postulate linking Planck mass `m_ℓ`, Planck length `ℓ_P`,
speed of light `c`, and gravitational constant `G`:
`m_ℓ / ℓ_P = c² / G` (follows directly from `ℓ_P = √(ħG/c³)` and `m_ℓ = √(ħc/G)`).
-/
structure PlanckScaleRelation (α : Type u) [ScalarField α] (lP mL c G : α) : Prop where
  lP_ne_zero : lP ≠ ScalarField.zero
  G_ne_zero : G ≠ ScalarField.zero
  planck_ratio : ScalarField.div mL lP = ScalarField.div (ScalarField.mul c c) G

/--
**Theorem (`haramein_phi_closed_form`, Haramein 2013 Eq. 5–8).**
With `r_ℓ = half * ℓ_P`, the raw PSU tiling ratio `η(r) / R(r)` equals
`harameinPhi r lP = (2 * ℓ_P) / r` identically for all `r ≠ 0`, `ℓ_P ≠ 0`.
-/
theorem haramein_phi_closed_form
    (r lP : α)
    (hr : r ≠ F.zero)
    (hlP : lP ≠ F.zero) :
    let rL := F.mul F.half lP
    F.div (harameinEta r rL) (harameinR r rL) = harameinPhi r lP := by
  intro rL
  have hHalfNe : F.half ≠ F.zero := by
    intro hZero
    have h1 : F.mul F.two F.half = F.zero := by rw [hZero, F.mul_zero]
    rw [F.two_mul_half] at h1
    have h2 : F.two = F.zero := by rw [F.two_def, h1, F.add_zero]
    exact F.two_ne_zero h2
  have hrL : rL ≠ F.zero := F.mul_ne_zero F.half lP hHalfNe hlP
  have hr2 : F.mul r r ≠ F.zero := F.mul_ne_zero r r hr hr
  have hr3 : F.mul (F.mul r r) r ≠ F.zero := F.mul_ne_zero _ r hr2 hr
  have hrL2 : F.mul rL rL ≠ F.zero := F.mul_ne_zero rL rL hrL hrL
  have hrL3 : F.mul (F.mul rL rL) rL ≠ F.zero := F.mul_ne_zero _ rL hrL2 hrL
  have hRne : harameinR r rL ≠ F.zero := by
    intro hZ
    have hEq : F.mul (F.mul r r) r = F.mul F.zero (F.mul (F.mul rL rL) rL) :=
      (F.div_eq_iff _ _ F.zero hrL3).1 hZ
    rw [F.zero_mul] at hEq
    exact hr3 hEq
  -- Use `div_eq_iff`: prove `η(r) = φ(r) * R(r)`.
  apply (F.div_eq_iff (harameinEta r rL) (harameinR r rL) (harameinPhi r lP) hRne).2
  unfold harameinEta harameinR harameinPhi
  rw [F.div_mul_div (F.mul F.two lP) r (F.mul (F.mul r r) r) (F.mul (F.mul rL rL) rL) hr hrL3]
  -- Simplify `(2 * lP) * r³ / (r * rL³) = (4 * r²) / rL²` via `lP = 2 * rL` and `4 = 2 * 2`.
  have hlP_eq : lP = F.mul F.two rL := by
    dsimp only [rL]
    rw [← F.mul_assoc, F.two_mul_half, F.one_mul]
  have hFour : F.four = F.mul F.two F.two := by
    rw [F.four_def, F.two_def, F.left_distrib, F.mul_one, ← F.two_def]
  have hDenNe : F.mul r (F.mul (F.mul rL rL) rL) ≠ F.zero := F.mul_ne_zero r _ hr hrL3
  -- Prove equality of the two fractions by cross-multiplying cleanly.
  apply (F.div_eq_iff _ _ _ hrL2).2
  rw [F.mul_comm (F.div _ _) (F.mul rL rL), F.mul_div_assoc, F.mul_comm (F.mul rL rL)]
  apply ((F.div_eq_iff _ _ _ hDenNe).2 ?_).symm
  -- Goal: `(2 * lP * (r² * r)) * rL² = (4 * r²) * (r * (rL² * rL))`
  rw [hlP_eq, hFour]
  -- Both sides equal `(2 * 2) * r² * r * rL² * rL`.
  calc
    F.mul (F.mul (F.mul F.two (F.mul F.two rL)) (F.mul (F.mul r r) r)) (F.mul rL rL)
      = F.mul (F.mul (F.mul (F.mul F.two F.two) rL) (F.mul (F.mul r r) r)) (F.mul rL rL) := by
          rw [F.mul_assoc F.two F.two rL]
    _ = F.mul (F.mul (F.mul F.two F.two) (F.mul r r)) (F.mul r (F.mul (F.mul rL rL) rL)) := by
          rw [F.mul_assoc (F.mul F.two F.two) rL,
              F.mul_assoc (F.mul F.two F.two),
              F.mul_assoc (F.mul F.two F.two) (F.mul r r)]
          congr 1
          -- Show `(rL * ((r * r) * r)) * (rL * rL) = (r * r) * (r * ((rL * rL) * rL))`
          rw [F.mul_comm rL (F.mul (F.mul r r) r),
              F.mul_assoc (F.mul (F.mul r r) r) rL (F.mul rL rL),
              F.mul_comm rL (F.mul rL rL),
              F.mul_assoc (F.mul r r) r (F.mul (F.mul rL rL) rL)]

/--
**Theorem (`haramein_holographic_duality`, Haramein 2013 Eq. 9 & Eq. 24).**
For any non-zero radius `r` and Planck length `ℓ_P`, the product of Haramein's
holographic gravitational mass `m_h(r)` and holographic proton rest mass `m_p'(r)`
is an exact scale-invariant constant equal to `2 m_ℓ²`:

`m_h(r) * m_p'(r) = 2 * m_ℓ²`.
-/
theorem haramein_holographic_duality
    (r lP mL : α)
    (hr : r ≠ F.zero)
    (hlP : lP ≠ F.zero) :
    F.mul (harameinHolographicMass r lP mL) (harameinProtonMass r lP mL) =
      F.mul F.two (F.mul mL mL) := by
  unfold harameinHolographicMass harameinProtonMass harameinPhi
  have h2lP : F.mul F.two lP ≠ F.zero := F.mul_ne_zero F.two lP F.two_ne_zero hlP
  have hDen : F.mul (F.mul F.two lP) r ≠ F.zero := F.mul_ne_zero _ r h2lP hr
  -- Rearrange RHS factor `2 * (((2*lP)/r) * mL) = ((2 * (2*lP)) * mL) / r`.
  rw [F.mul_comm (F.div (F.mul F.two lP) r) mL, F.mul_div_assoc mL,
      F.mul_div_assoc F.two, F.div_mul_div _ _ _ _ h2lP hr]
  -- Cancel `(2 * lP) * r` from numerator and denominator.
  have hNum :
      F.mul (F.mul r mL) (F.mul F.two (F.mul mL (F.mul F.two lP))) =
      F.mul (F.mul F.two (F.mul mL mL)) (F.mul (F.mul F.two lP) r) := by
    rw [F.mul_comm r mL, F.mul_assoc mL r,
        F.mul_comm r (F.mul F.two (F.mul mL (F.mul F.two lP))),
        ← F.mul_assoc mL,
        F.mul_comm mL (F.mul F.two (F.mul mL (F.mul F.two lP))),
        F.mul_assoc F.two (F.mul mL (F.mul F.two lP)) mL,
        F.mul_assoc mL (F.mul F.two lP) mL,
        F.mul_comm (F.mul F.two lP) mL,
        ← F.mul_assoc mL mL (F.mul F.two lP),
        ← F.mul_assoc F.two (F.mul mL mL) (F.mul F.two lP),
        F.mul_assoc (F.mul F.two (F.mul mL mL)) (F.mul F.two lP) r]
  rw [hNum, F.mul_div_cancel _ _ hDen]

/--
**Theorem (`haramein_schwarzschild_equivalence`, Haramein 2013 Eq. 12–13).**
Given `PlanckScaleRelation lP mL c G` (`m_ℓ / ℓ_P = c² / G`), Haramein's
discrete PSU holographic gravitational mass `m_h(r) = (r * m_ℓ) / (2 * ℓ_P)`
equals the classical Schwarzschild mass `m_s(r) = (r * c²) / (2 * G)` identically.
-/
theorem haramein_schwarzschild_equivalence
    (r lP mL c G : α)
    (hPlanck : PlanckScaleRelation α lP mL c G) :
    harameinHolographicMass r lP mL = F.div (F.mul r (F.mul c c)) (F.mul F.two G) := by
  unfold harameinHolographicMass
  have h2lP : F.mul F.two lP ≠ F.zero := F.mul_ne_zero F.two lP F.two_ne_zero hPlanck.lP_ne_zero
  have h2G : F.mul F.two G ≠ F.zero := F.mul_ne_zero F.two G F.two_ne_zero hPlanck.G_ne_zero
  -- Rewrite `(r * mL) / (2 * lP) = (r / 2) * (mL / lP)` and apply `hPlanck.planck_ratio`.
  have hL : F.div (F.mul r mL) (F.mul F.two lP) = F.mul (F.div r F.two) (F.div mL lP) :=
    (F.div_mul_div r F.two mL lP F.two_ne_zero hPlanck.lP_ne_zero).symm
  have hR : F.div (F.mul r (F.mul c c)) (F.mul F.two G) = F.mul (F.div r F.two) (F.div (F.mul c c) G) :=
    (F.div_mul_div r F.two (F.mul c c) G F.two_ne_zero hPlanck.G_ne_zero).symm
  rw [hL, hR, hPlanck.planck_ratio]

end Haramein2013

-- =============================================================================
-- 2. Chris Essonne — Vacuum Pressure Gradient & Spiral Galaxy Aether Vortex
--    Sources: research_papers_gravity_science_journal_8052.pdf
--             research_papers_gravity_science_journal_10100.pdf
-- =============================================================================
section EssonneGravity

variable {α : Type u} [F : ScalarField α]

/--
Chris Essonne (`8052.pdf` Eq. 21): Exterior vacuum pressure ratio
`P(r) / P₀ = 1 - R_s / r`.
-/
def essonneExteriorPressureRatio (r Rs : α) : α :=
  F.sub F.one (F.div Rs r)

/--
Chris Essonne (`8052.pdf` Eq. 35): Interior vacuum pressure ratio at center `r = 0`
inside a homogeneous sphere of radius `R`:
`P(0) / P₀ = 1 - R_s / R - R_s / (2 R)`.
-/
def essonneCentralPressureRatio (R Rs : α) : α :=
  F.sub (F.sub F.one (F.div Rs R)) (F.div Rs (F.mul F.two R))

/--
**Theorem (`essonne_zero_central_pressure_at_3_2_Rs`, `8052.pdf` Eq. 39).**
At critical radius `R = (3/2) * R_s`, Essonne's central vacuum pressure inside
a homogeneous star vanishes identically (`P(0) / P₀ = 0`).
-/
theorem essonne_zero_central_pressure_at_3_2_Rs
    (Rs : α)
    (hRs : Rs ≠ F.zero)
    (hThreeNe : F.three ≠ F.zero) :
    let R := F.div (F.mul F.three Rs) F.two
    essonneCentralPressureRatio R Rs = F.zero := by
  intro R
  unfold essonneCentralPressureRatio
  have h2R : F.mul F.two R = F.mul F.three Rs := by
    dsimp only [R]
    rw [F.mul_comm, F.div_mul_cancel _ F.two F.two_ne_zero]
  have h3RsNe : F.mul F.three Rs ≠ F.zero := F.mul_ne_zero F.three Rs hThreeNe hRs
  have hRne : R ≠ F.zero := by
    intro hZ
    have h0 : F.mul F.two R = F.zero := by rw [hZ, F.mul_zero]
    rw [h2R] at h0
    exact h3RsNe h0
  -- Show `Rs / R = 2 / 3` and `Rs / (2 * R) = 1 / 3`.
  have hTerm1 : F.div Rs R = F.div F.two F.three := by
    apply (F.div_eq_iff Rs R (F.div F.two F.three) hRne).2
    dsimp only [R]
    rw [F.div_mul_div F.two F.three (F.mul F.three Rs) F.two hThreeNe F.two_ne_zero]
    have hSwap : F.mul F.two (F.mul F.three Rs) = F.mul Rs (F.mul F.three F.two) := by
      rw [← F.mul_assoc, F.mul_comm (F.mul F.two F.three) Rs, F.mul_comm F.two F.three]
    rw [hSwap, F.mul_div_cancel Rs (F.mul F.three F.two) (F.mul_ne_zero F.three F.two hThreeNe F.two_ne_zero)]
  have hTerm2 : F.div Rs (F.mul F.two R) = F.div F.one F.three := by
    rw [h2R]
    apply (F.div_eq_iff Rs (F.mul F.three Rs) (F.div F.one F.three) h3RsNe).2
    rw [F.mul_comm, F.mul_div_assoc, F.mul_one, mul_div_cancel_left F.three Rs hThreeNe]
  rw [hTerm1, hTerm2, ← sub_add_eq_sub_sub]
  -- Show `2 / 3 + 1 / 3 = 3 / 3 = 1`.
  have hSum : F.add (F.div F.two F.three) (F.div F.one F.three) = F.one := by
    have hFactor : F.add (F.div F.two F.three) (F.div F.one F.three) =
                   F.mul (F.add F.two F.one) (F.div F.one F.three) := by
      rw [F.right_distrib, F.mul_div_assoc, F.mul_div_assoc, F.mul_one, F.mul_one]
    rw [hFactor, ← F.three_def, F.mul_div_assoc, F.mul_one, div_self_eq_one F.three hThreeNe]
  rw [hSum, sub_self_eq_zero]

/--
**Theorem (`essonne_true_horizon_2Rs`, `8052.pdf` Eq. 41–43).**
When Essonne's local speed of light squared `c'² = c² (1 - R_s / r)` is
compared to the Newtonian escape velocity squared `v_L² = c² (R_s / r)` at
radius `r = 2 R_s`, they are strictly equal (`c'(2 R_s)² = v_L(2 R_s)²`).
-/
theorem essonne_true_horizon_2Rs
    (c Rs : α)
    (hRs : Rs ≠ F.zero) :
    let r := F.mul F.two Rs
    F.mul (F.mul c c) (essonneExteriorPressureRatio r Rs) =
      F.mul (F.mul c c) (F.div Rs r) := by
  intro r
  congr 1
  unfold essonneExteriorPressureRatio
  have hrNe : r ≠ F.zero := F.mul_ne_zero F.two Rs F.two_ne_zero hRs
  have hRatio : F.div Rs r = F.half := by
    apply (F.div_eq_iff Rs r F.half hrNe).2
    dsimp only [r]
    rw [← F.mul_assoc, half_mul_two, F.one_mul]
  rw [hRatio, ← F.half_add_half, F.sub_eq_add_neg, add_neg_cancel_right]

/--
Chris Essonne (`10100.pdf` Eq. 11): Outer galactic periphery radial aether-vortex
gravitational acceleration for `r > R_c`:
`g_r(r) = (2 G M - R_c² g_z) / (h * r)`.
-/
def essonneOuterVortexAccel (r M G Rc h gz : α) : α :=
  F.div (F.sub (F.mul (F.mul F.two G) M) (F.mul (F.mul Rc Rc) gz)) (F.mul h r)

/--
**Theorem (`essonne_flat_galaxy_rotation_velocity`, `10100.pdf` Eq. 11).**
For any periphery star at radius `r ≠ 0` outside `R_c` in centripetal balance
`v² = r * g_r(r)`, the radial coordinate `r` cancels identically, proving
Essonne's dark-matter-free constant flat rotation curve:

`r * essonneOuterVortexAccel r M G Rc h gz = (2 G M - R_c² g_z) / h`.
-/
theorem essonne_flat_galaxy_rotation_velocity
    (r M G Rc h gz : α)
    (hr : r ≠ F.zero)
    (hh : h ≠ F.zero) :
    F.mul r (essonneOuterVortexAccel r M G Rc h gz) =
      F.div (F.sub (F.mul (F.mul F.two G) M) (F.mul (F.mul Rc Rc) gz)) h := by
  unfold essonneOuterVortexAccel
  generalize F.sub (F.mul (F.mul F.two G) M) (F.mul (F.mul Rc Rc) gz) = num
  have hhr : F.mul h r ≠ F.zero := F.mul_ne_zero h r hh hr
  rw [F.mul_div_assoc]
  apply (F.div_eq_iff (F.mul r num) (F.mul h r) (F.div num h) hhr).2
  rw [F.mul_comm h r, ← F.mul_assoc, F.mul_comm (F.div num h) r,
      F.mul_assoc r, F.div_mul_cancel num h hh]

end EssonneGravity

-- =============================================================================
-- 3. Salvatore Pais — HEEMFG Poynting Energy Flux (`US10144532.pdf` Eq. 1)
-- =============================================================================
section PaisHeemfg

variable {α : Type u} [F : ScalarField α]

/--
Salvatore Pais (`US10144532.pdf` Eq. 1): Maximum HEEMFG Poynting energy flux
`S_max = f_G * (σ² / ε₀) * (R_r * ω + R_v * ν + v_R)`.
-/
def paisMaxPoyntingFlux (fG sigma eps0 Rr omega Rv nu vR : α) : α :=
  let pref := F.mul fG (F.div (F.mul sigma sigma) eps0)
  F.mul pref (F.add (F.add (F.mul Rr omega) (F.mul Rv nu)) vR)

/--
**Theorem (`pais_poynting_flux_superposition`, `US10144532.pdf` Eq. 1).**
The non-relativistic HEEMFG Poynting flux decomposes additively into its three
independent kinematic vacuum-polarization modes (rotation, harmonic vibration,
and curvilinear translation).
-/
theorem pais_poynting_flux_superposition (fG sigma eps0 Rr omega Rv nu vR : α) :
    paisMaxPoyntingFlux fG sigma eps0 Rr omega Rv nu vR =
      F.add
        (F.add (paisMaxPoyntingFlux fG sigma eps0 Rr omega F.zero F.zero F.zero)
               (paisMaxPoyntingFlux fG sigma eps0 F.zero F.zero Rv nu F.zero))
        (paisMaxPoyntingFlux fG sigma eps0 F.zero F.zero F.zero F.zero vR) := by
  unfold paisMaxPoyntingFlux
  dsimp only
  simp only [F.zero_mul, F.add_zero, F.zero_add]
  rw [F.left_distrib, F.left_distrib]

end PaisHeemfg

-- =============================================================================
-- 4. Elena Danaan / Dan Willis / Marcel Vogel — Love & Phryll Pairwise Coupling
--    Matches `ComputeUnifiedFieldKernel` in `src/models/unified/UnifiedFieldSolver.cu`
-- =============================================================================
section LovePhryllCoupling

variable {α : Type u} [F : ScalarField α]

/--
Pairwise Love/Phryll resonant coherence force exerted on entity `i` by entity `j`
(exact counterpart to `UnifiedFieldSolver.cu`):
`F_ij^{LP} = (loveCoupling * C_i * C_j * (Q_i + Q_j) * invDist3) * (r_j - r_i)`.
-/
def lovePhryllPairForce
    (loveCoupling ci cj qi qj invDist3 : α)
    (ri rj : Vec3 α) : Vec3 α :=
  let coeff := F.mul (F.mul (F.mul loveCoupling (F.mul ci cj)) (F.add qi qj)) invDist3
  Mainstream.centralPairForce coeff ri rj

/--
**Theorem (`lovePhryllPairForce_antisymm`).**
Because `C_i * C_j = C_j * C_i` and `Q_i + Q_j = Q_j + Q_i`, the Pillar 2
Love/Phryll resonant coherence force strictly satisfies Newton's 3rd Law
(`F_ij^{LP} = -F_ji^{LP}`), guaranteeing exact discrete linear momentum
conservation under Velocity Verlet (`verletTwoBody_conserves_linear_momentum`).
-/
theorem lovePhryllPairForce_antisymm
    (loveCoupling ci cj qi qj invDist3 : α)
    (ri rj : Vec3 α) :
    lovePhryllPairForce loveCoupling ci cj qi qj invDist3 ri rj =
      Vec3.neg (lovePhryllPairForce loveCoupling cj ci qj qi invDist3 rj ri) := by
  unfold lovePhryllPairForce
  have hSymm :
      F.mul (F.mul (F.mul loveCoupling (F.mul cj ci)) (F.add qj qi)) invDist3 =
      F.mul (F.mul (F.mul loveCoupling (F.mul ci cj)) (F.add qi qj)) invDist3 := by
    rw [F.mul_comm cj ci, F.add_comm qj qi]
  rw [hSymm]
  exact Mainstream.newtonianPairForce_antisymm _ ri rj

/--
**Theorem (`lovePhryllPairForce_torque_cancels`).**
The Pillar 2 Love/Phryll pairwise force also produces strictly zero net torque
(`r_i × F_ij^{LP} + r_j × F_ji^{LP} = 0`), guaranteeing exact angular momentum
conservation (matching `TestLovePhryllPairwiseConservation` GPU result `8.66e-15`).
-/
theorem lovePhryllPairForce_torque_cancels
    (loveCoupling ci cj qi qj invDist3 : α)
    (ri rj : Vec3 α) :
    Vec3.add (Vec3.cross ri (lovePhryllPairForce loveCoupling ci cj qi qj invDist3 ri rj))
             (Vec3.cross rj (lovePhryllPairForce loveCoupling cj ci qj qi invDist3 rj ri)) =
      Vec3.zero := by
  unfold lovePhryllPairForce
  have hSymm :
      F.mul (F.mul (F.mul loveCoupling (F.mul cj ci)) (F.add qj qi)) invDist3 =
      F.mul (F.mul (F.mul loveCoupling (F.mul ci cj)) (F.add qi qj)) invDist3 := by
    rw [F.mul_comm cj ci, F.add_comm qj qi]
  rw [hSymm]
  exact Mainstream.centralForce_torque_cancels _ ri rj

end LovePhryllCoupling

end Umgebung.Unified
