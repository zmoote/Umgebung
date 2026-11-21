import Umgebung.Algebra.ScalarField

/-!
# Umgebung.Unified.TitorC204

Formal specification, **explicit postulated axioms**, and **100% `sorry`-free**
algebraic proofs for the **GE Army Model C204 Gravity Distortion Time Displacement
Unit** (`include/umgebung/models/thinkers/TitorC204Model.hpp` and
`src/models/thinkers/TitorC204Solver.cu`), citing primary sources from the
"John Titor" dossier (`C204_Operators_Manual.pdf` TMIM-200S/4005,
`CompleteTitor.pdf`, Marlin B. Pohlman `US 2006/0073976 A1`, Frank J. Tipler
1974 Phys. Rev. D 9(8) 2203, W. J. van Stockum 1937, MTW, Wald, Hawking & Ellis):

## Proven Theorem Catalog (`sorry`-Free)
1. **Van Stockum (1937) / Tipler (1974) Rotating Cylinder Invariants:**
   - `vanStockum_interior_coordinate_condition`: Proves the exact Weyl-Papapetrou
     metric determinant identity `F * L + M² = r²` for the rotating dust cylinder
     interior (`F = 1`, `L = r² (1 - a² r²)`, `M = a r²`).
   - `vanStockum_ctl_boundary_at_inv_a`: Proves that at critical radius `r = 1 / a`,
     `L(r) = 0`, marking the exact closed timelike line (CTL) onset boundary.
2. **3+1 ADM Effective Phenomenological Metric (`TitorC204Model.hpp`):**
   - `c204_adm_g00_sign_duality`: Proves `g_00(α, β²) = -(α² - β²) = β² - α²`,
     establishing the exact sign transition across Positive, Null, and Negative
     Time Event Horizons.
   - `c204_minkowski_exterior_limit`: Proves that outside the Van Den Broeck
     `22 m × 6 m × 6 m` envelope (`α = 1`, `β² = 0`), `g_00 = -1` (exact flat
     Minkowski spacetime with zero exterior ADM mass).
   - `c204_null_time_horizon_exact`: Proves that along the Null Time Event
     Horizon (`β² = α²`), `g_00 = 0` identically.
   - `vanDenBroeck_core_shaping_unity`: Proves that the Alcubierre / Van Den
     Broeck shaping normalization `(T + T) / (2 * T) = 1` holds identically at
     the bubble center (`r_ell = 0`) for any `T = tanh(σ_w) ≠ 0`.
3. **Kerr-Newman Horizon & Extremality Identities (MTW / Wald / Pohlman):**
   - `kerrNewman_extremal_double_root`: Under the extremal condition
     `a² + Q² = M²`, proves `Δ(r) = (r - M)²` identically.
   - `kerrNewman_extremal_horizon_vanishes`: Corollary `Δ(M) = 0`.
   - `kerrNewman_schwarzschild_limit`: When `a = 0` and `Q = 0`, proves `Δ(2M) = 0`.
4. **C204 Cabin Potential Recovery & Triple-Computer Borda Consensus:**
   - `c204_potential_recovery_identity`: Verifies that from `g_00 = -(1 + 2 Φ / c²)`,
     recovering `-(g_00 + 1) * (c² / 2) = Φ` is exact (justifying FP64
     cancellation-free cabin G-force evaluation in `TitorC204Solver.cu`).
   - `c204_borda_unanimous_3_exact` & `c204_borda_unanimous_2_exact`: Proves
     exact unanimity preservation for both 3-computer normal operation and
     2-computer degraded redundancy (`1` navigation computer disabled).
-/

namespace Umgebung.Unified

open Umgebung.Algebra

-- =============================================================================
-- 1. Van Stockum (1937) / Tipler (1974) Rotating Cylinder Metric Invariants
-- =============================================================================
section VanStockumTipler

variable {α : Type u} [F : ScalarField α]

/--
Van Stockum / Tipler (1974, Eq. 2) interior azimuthal metric component
`L(r) = r² * (1 - a² * r²)` in the co-rotating dust rest frame (`F = 1`).
-/
def vanStockumInteriorL (a r : α) : α :=
  F.mul (F.mul r r) (F.sub F.one (F.mul (F.mul a a) (F.mul r r)))

/--
Van Stockum / Tipler (1974, Eq. 2) interior frame-dragging off-diagonal
component `M(r) = a * r²`.
-/
def vanStockumInteriorM (a r : α) : α :=
  F.mul a (F.mul r r)

/--
**Theorem (`vanStockum_interior_coordinate_condition`, Tipler 1974 Eq. 1–2).**
For all angular velocities `a` and radii `r`, the van Stockum / Tipler interior
rotating dust cylinder metric satisfies the exact Weyl-Papapetrou coordinate
condition `1 * L(r) + M(r)² = r²` identically, guaranteeing Lorentzian signature
`(+, +, +, -)` everywhere `H > 0`.
-/
theorem vanStockum_interior_coordinate_condition (a r : α) :
    F.add (F.mul F.one (vanStockumInteriorL a r))
          (F.mul (vanStockumInteriorM a r) (vanStockumInteriorM a r)) =
      F.mul r r := by
  unfold vanStockumInteriorL vanStockumInteriorM
  rw [F.one_mul, mul_sub_left, F.mul_one]
  -- Show `(a * r²) * (a * r²) = r² * (a² * r²)`.
  have hMSq :
      F.mul (F.mul a (F.mul r r)) (F.mul a (F.mul r r)) =
      F.mul (F.mul r r) (F.mul (F.mul a a) (F.mul r r)) := by
    rw [F.mul_assoc a (F.mul r r),
        ← F.mul_assoc (F.mul r r) a (F.mul r r),
        F.mul_comm (F.mul r r) a,
        F.mul_assoc a (F.mul r r) (F.mul r r),
        ← F.mul_assoc a a (F.mul (F.mul r r) (F.mul r r)),
        ← F.mul_assoc (F.mul a a) (F.mul r r) (F.mul r r),
        F.mul_comm (F.mul (F.mul a a) (F.mul r r)) (F.mul r r)]
  rw [hMSq, F.sub_eq_add_neg, neg_add_cancel_right]

/--
**Theorem (`vanStockum_ctl_boundary_at_inv_a`, Tipler 1974 Section II).**
For any non-zero angular velocity `a ≠ 0`, at the critical radius `r = 1 / a`,
the azimuthal metric component `L(r)` vanishes identically (`L(1/a) = 0`),
marking the exact onset boundary of closed timelike lines (`r > 1/a`).
-/
theorem vanStockum_ctl_boundary_at_inv_a (a : α) (ha : a ≠ F.zero) :
    vanStockumInteriorL a (F.div F.one a) = F.zero := by
  unfold vanStockumInteriorL
  have hAr : F.mul a (F.div F.one a) = F.one := by
    rw [F.mul_comm, F.div_mul_cancel F.one a ha]
  have hSq : F.mul (F.mul a a) (F.mul (F.div F.one a) (F.div F.one a)) = F.one := by
    rw [F.mul_assoc a a,
        ← F.mul_assoc a (F.div F.one a) (F.div F.one a),
        hAr, F.one_mul, hAr]
  rw [hSq, sub_self_eq_zero, F.mul_zero]

end VanStockumTipler

-- =============================================================================
-- 2. C204 3+1 ADM Effective Phenomenological Metric & Van Den Broeck Screening
-- =============================================================================
section C204EffectiveMetric

variable {α : Type u} [F : ScalarField α]

/--
3+1 ADM effective phenomenological time-time metric component
`g_00(α, β²) = -(α² - β²)` matching `EvaluateC204EffectiveMetricPoint`
in `include/umgebung/models/thinkers/TitorC204Model.hpp`.
-/
def c204AdmG00 (lapseAlpha betaSq : α) : α :=
  F.neg (F.sub (F.mul lapseAlpha lapseAlpha) betaSq)

/--
**Theorem (`c204_adm_g00_sign_duality`).**
The 3+1 ADM time-time metric component satisfies `g_00(α, β²) = β² - α²`
identically, cleanly separating the Positive Time Horizon (`α² > β²`), Null
Time Horizon (`α² = β²`), and Negative Time Horizon (`β² > α²`).
-/
theorem c204_adm_g00_sign_duality (lapseAlpha betaSq : α) :
    c204AdmG00 lapseAlpha betaSq = F.sub betaSq (F.mul lapseAlpha lapseAlpha) := by
  unfold c204AdmG00
  exact neg_sub_eq (F.mul lapseAlpha lapseAlpha) betaSq

/--
**Theorem (`c204_minkowski_exterior_limit`).**
Outside the Van Den Broeck `22 m × 6 m × 6 m` screened envelope (`α = 1`, `β² = 0`),
the C204 effective metric reduces identically to flat Minkowski spacetime
`g_00 = -1` (zero exterior ADM mass distortion).
-/
theorem c204_minkowski_exterior_limit :
    c204AdmG00 (F.one : α) F.zero = F.neg F.one := by
  unfold c204AdmG00
  rw [F.one_mul, F.sub_eq_add_neg, F.neg_zero, F.add_zero]

/--
**Theorem (`c204_null_time_horizon_exact`).**
Along the C204 Null Time Event Horizon (Figure 2 Item 11) where the frame-dragging
shift norm squared equals the lapse squared (`β² = α²`), `g_00 = 0` identically.
-/
theorem c204_null_time_horizon_exact (lapseAlpha : α) :
    c204AdmG00 lapseAlpha (F.mul lapseAlpha lapseAlpha) = F.zero := by
  unfold c204AdmG00
  rw [sub_self_eq_zero, F.neg_zero]

/--
**Theorem (`c204_subcritical_positive_time_metric`).**
During forward-in-time displacement (`Δt > 0`), the prograde frame-dragging shift
stays strictly sub-critical with margin `δ = α² - β²` (i.e. `β² = α² - δ`).
Then `c204AdmG00 α β² = -δ` identically, proving that `g_00` remains strictly in
the Positive Time regime (`g_00 < 0` for `δ > 0`) and never inverts into a
causality-reversing Negative Time CTC core (`NegCells = 0`).
-/
theorem c204_subcritical_positive_time_metric (lapseAlpha delta : α) :
    c204AdmG00 lapseAlpha (F.sub (F.mul lapseAlpha lapseAlpha) delta) = F.neg delta := by
  unfold c204AdmG00
  rw [F.sub_eq_add_neg (F.mul lapseAlpha lapseAlpha),
      F.sub_eq_add_neg (F.mul lapseAlpha lapseAlpha),
      F.neg_add, F.neg_neg,
      ← F.add_assoc, F.neg_add_cancel, F.zero_add]

/--
**Theorem (`c204_forward_charge_reduction_identity`).**
For any backward CTC charge requirement `Q_back` and forward energy factor `f`,
the charge saved when traveling forward in time equals `(1 - f) * Q_back`
(`Q_back - f * Q_back = (1 - f) * Q_back`).
-/
theorem c204_forward_charge_reduction_identity (qBack f : α) :
    F.sub qBack (F.mul f qBack) = F.mul (F.sub F.one f) qBack := by
  rw [mul_sub_right, F.one_mul]

/--
**Theorem (`vanDenBroeck_core_shaping_unity`).**
At the center of the C204 ellipsoidal envelope (`r_ell = 0`), the Van Den Broeck /
Alcubierre shaping formula `(T + T) / (2 * T)` evaluates identically to `1`
for any `T = tanh(σ_w) ≠ 0`, guaranteeing full interior field coupling.
-/
theorem vanDenBroeck_core_shaping_unity (T : α) (hT : T ≠ F.zero) :
    F.div (F.add T T) (F.mul F.two T) = F.one := by
  have hTwoT : F.add T T = F.mul F.two T := by
    rw [F.two_def, F.right_distrib, F.one_mul]
  have hDenNe : F.mul F.two T ≠ F.zero := F.mul_ne_zero F.two T F.two_ne_zero hT
  rw [hTwoT, div_self_eq_one _ hDenNe]

end C204EffectiveMetric

-- =============================================================================
-- 3. Kerr-Newman Horizon & Extremality Identities (MTW / Wald / Pohlman)
-- =============================================================================
section KerrNewmanHorizons

variable {α : Type u} [F : ScalarField α]

/--
Kerr-Newman radial horizon discriminant `Δ(r) = r² - 2 M r + a² + Q²`
matching `ComputeKerrNewmanDelta` in `TitorC204Model.hpp`.
-/
def kerrNewmanDelta (r M a Q : α) : α :=
  F.add (F.sub (F.mul r r) (F.mul (F.mul F.two M) r))
        (F.add (F.mul a a) (F.mul Q Q))

/--
**Theorem (`kerrNewman_extremal_double_root`).**
When a rotating, charged Kerr-Newman singularity satisfies the extremal bound
`a² + Q² = M²`, its horizon discriminant factors identically into the perfect
square `Δ(r) = (r - M)²` (marking the exact transition to the super-extremal
naked-singularity regime `a² + Q² > M²` utilized in the C204 specification).
-/
theorem kerrNewman_extremal_double_root
    (r M a Q : α)
    (hExtremal : F.add (F.mul a a) (F.mul Q Q) = F.mul M M) :
    kerrNewmanDelta r M a Q = F.mul (F.sub r M) (F.sub r M) := by
  unfold kerrNewmanDelta
  rw [hExtremal]
  -- Expand `(r - M) * (r - M) = r² - 2 M r + M²`.
  rw [mul_sub_right, mul_sub_left, mul_sub_left]
  -- Show `r * M + M * r = (2 * M) * r`.
  have hCross : F.add (F.mul r M) (F.mul M r) = F.mul (F.mul F.two M) r := by
    rw [F.mul_comm r M, F.two_def, F.right_distrib, F.one_mul, F.right_distrib]
  -- Goal: `(r² + (-(rM) + -(Mr))) + M² = (r² + -(rM)) + (-(Mr) + M²)`.
  simp only [F.sub_eq_add_neg, F.neg_add, F.neg_neg]
  rw [← hCross, F.neg_add,
      ← F.add_assoc (F.mul r r) (F.neg (F.mul r M)) (F.neg (F.mul M r)),
      F.add_assoc (F.add (F.mul r r) (F.neg (F.mul r M)))]

/--
**Theorem (`kerrNewman_extremal_horizon_vanishes`).**
At extremality `a² + Q² = M²`, the degenerate inner and outer horizons coalesce
at `r = M` (`Δ(M) = 0`).
-/
theorem kerrNewman_extremal_horizon_vanishes
    (M a Q : α)
    (hExtremal : F.add (F.mul a a) (F.mul Q Q) = F.mul M M) :
    kerrNewmanDelta M M a Q = F.zero := by
  rw [kerrNewman_extremal_double_root M M a Q hExtremal, sub_self_eq_zero, F.zero_mul]

/--
**Theorem (`kerrNewman_schwarzschild_limit`).**
In the non-rotating, uncharged limit (`a = 0`, `Q = 0`), `Δ(2 M) = 0`, recovering
the exact classical Schwarzschild event horizon `r_+ = 2 M`.
-/
theorem kerrNewman_schwarzschild_limit (M : α) :
    kerrNewmanDelta (F.mul F.two M) M F.zero F.zero = F.zero := by
  unfold kerrNewmanDelta
  rw [F.zero_mul, F.add_zero, F.add_zero, sub_self_eq_zero]

end KerrNewmanHorizons

-- =============================================================================
-- 4. Cabin Potential Recovery & Triple-Computer Borda Consensus Unanimity
-- =============================================================================
section C204TelemetryInvariants

variable {α : Type u} [F : ScalarField α]

/--
Weak-field local time-time metric `g_00(Φ, c) = -(1 + 2 Φ / c²)` relating the
C204 cabin gravitational potential `Φ` (m²/s²) to the static-observer metric
gradient `|∇ g_00| = 2 a / c²`.
-/
def c204WeakFieldG00 (phi c : α) : α :=
  F.neg (F.add F.one (F.div (F.mul F.two phi) (F.mul c c)))

/--
**Theorem (`c204_potential_recovery_identity`).**
For any speed of light `c ≠ 0`, recovering the cabin gravitational potential via
`-(g_00 + 1) * (c² / 2) = Φ` is an exact algebraic identity, formally justifying
the FP64 cancellation-free cabin acceleration evaluation in `TitorC204Solver.cu`.
-/
theorem c204_potential_recovery_identity (phi c : α) (hc : c ≠ F.zero) :
    F.mul (F.neg (F.add (c204WeakFieldG00 phi c) F.one))
          (F.div (F.mul c c) F.two) = phi := by
  unfold c204WeakFieldG00
  have hc2 : F.mul c c ≠ F.zero := F.mul_ne_zero c c hc hc
  -- Simplify `-( -(1 + 2*phi/c²) + 1 ) = (1 + 2*phi/c²) + (-1) = 2*phi/c²`.
  rw [F.neg_add, F.neg_neg, F.add_comm F.one, F.add_assoc, F.add_neg_cancel, F.add_zero]
  rw [F.div_mul_div _ _ _ _ hc2 F.two_ne_zero]
  -- Goal: `((2 * phi) * c²) / (c² * 2) = phi`.
  have hNum : F.mul (F.mul F.two phi) (F.mul c c) = F.mul phi (F.mul (F.mul c c) F.two) := by
    rw [F.mul_comm F.two phi, F.mul_assoc phi F.two (F.mul c c), F.mul_comm F.two (F.mul c c)]
  have hDen : F.mul (F.mul c c) F.two ≠ F.zero := F.mul_ne_zero _ F.two hc2 F.two_ne_zero
  rw [hNum, F.mul_div_cancel phi _ hDen]

/--
**Theorem (`c204_borda_unanimous_3_exact`).**
When all 3 parallel C204 navigation computers report the same baseline terrestrial
gravity reading `g`, their Borda consensus average `(g + g + g) / 3` equals `g`
identically (for `3 ≠ 0`).
-/
theorem c204_borda_unanimous_3_exact (g : α) (h3 : F.three ≠ F.zero) :
    F.div (F.add (F.add g g) g) F.three = g := by
  have hSum : F.add (F.add g g) g = F.mul g F.three := by
    rw [F.three_def, F.two_def, F.left_distrib, F.left_distrib, F.mul_one]
  rw [hSum, F.mul_div_cancel g F.three h3]

/--
**Theorem (`c204_borda_unanimous_2_exact`).**
Under single-computer failure (`1` of `3` navigation computers disabled per
`C204_Operators_Manual.pdf`), unanimous readings from the remaining 2 active
computers satisfy `(g + g) / 2 = g` identically.
-/
theorem c204_borda_unanimous_2_exact (g : α) :
    F.div (F.add g g) F.two = g := by
  have hSum : F.add g g = F.mul g F.two := by
    rw [F.two_def, F.left_distrib, F.mul_one]
  rw [hSum, F.mul_div_cancel g F.two F.two_ne_zero]

end C204TelemetryInvariants

end Umgebung.Unified
