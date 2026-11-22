/**
 * @file TitorC204Model.hpp
 * @brief Verbatim operational specifications, foundational general-relativistic
 *        metrics, and calibrated 3+1 ADM **Effective Phenomenological Metric**
 *        for the **GE Army Model C204 Gravity Distortion Time Displacement Unit**
 *        ("John Titor" / TimeTravel_0, Marlin B. Pohlman `US 2006/0073976 A1`,
 *        Frank J. Tipler 1974, and W. J. van Stockum 1937).
 *
 * @details
 * Synthesizes every verbatim specification and theoretical framework cataloged
 * in the "John Titor" research dossier with strict epistemological transparency
 * separating verbatim hardware/textbook relations from calibrated
 * phenomenological ansatzes:
 *
 * ### 1. Verbatim Hardware & Operational Specifications (`C204_Operators_Manual.pdf`, `TMIM-200S/4005`)
 *   - **Unit Mass & Dimensions:** $225\,\text{kg}$ dry mass; $44\,\text{cm}$ (H) $\times$
 *     $48\,\text{cm}$ (W) $\times$ $140\,\text{cm}$ (L); max gross payload $2500\,\text{kg}$.
 *   - **Dual Positive Top-Spin Micro-Singularities:**
 *     - Drive Singularity: `VERT / OVER-UNDER` spin, `POSITIVE` charge.
 *     - Steering Singularity: `HORZ / OVER-UNDER` spin, `POSITIVE` charge.
 *     - Output: $300\text{--}500\,\text{MW}$ per electrified singularity; housed across
 *       the $48\,\text{cm}$ chassis width inside magnetic/vacuum containment.
 *   - **Centered Loading Envelope & Tipler Sinusoid Geometry (Figure 2):**
 *     - Ellipsoidal envelope: $22\,\text{m}$ length ($a = 11\,\text{m}$),
 *       $6\,\text{m}$ width ($b = 3\,\text{m}$), $6\,\text{m}$ height ($c = 3\,\text{m}$).
 *     - Vertical safe distance: $6\,\text{m}$; mass offset: $5\,\text{m}$;
 *       rear mass distance: $8\,\text{m}$; forward mass distance: $11\,\text{m}$
 *       (yielding longitudinal center offset $x_{\text{offset}} = +1.5\,\text{m}$).
 *     - Rearward X-ray venting cone: $170^\circ$ arc ($\pm 85^\circ$ around $-x$).
 *     - Concentric horizons: Negative Time Event Horizon (inner), Null Time Event
 *       Horizon (intermediate), and Positive Time Event Horizon (outer).
 *   - **Power Settings, Wave Penetration Rates & Electrical Charge Ranges:**
 *     - `100%` Power: $10.0\,\text{yrs/hr}$ ($v_t \approx 87{,}660\,\text{s/s}$), $288{,}000\,\text{C}$.
 *     - `75%`  Power: $6.2\,\text{yrs/hr}$  ($v_t \approx 54{,}349\,\text{s/s}$), $201{,}600\,\text{C}$.
 *     - `50%`  Power: $3.4\,\text{yrs/hr}$  ($v_t \approx 29{,}804\,\text{s/s}$), $126{,}000\,\text{C}$.
 *     - **VGL (Variable Gravity Lock):** Pulses every $240\,\text{s}$, $54{,}000\,\text{C}$ total.
 *   - **Clocks, Computing & Divergence Limits:**
 *     - $4$ Cesium clocks (`Cesium 14-1` main, `Cesium 10-1` backup), $1\,\text{s}$ increment.
 *     - $3$ parallel navigation computers linked by a **Borda error-correcting protocol**
 *       (fully operable with $1$ computer disabled); $10^{39}$ calculated destinations.
 *     - Accuracy break-away horizon: $60\,\text{yrs}$; nominal divergence $\approx 1\%$ per
 *       $60\,\text{yrs}$ at max power ($2.5\%$ observed on $2036 \to 1975$ after $2$ VGL checks).
 *   - **Physical & Environmental Telemetry (`CompleteTitor.pdf`):**
 *     - Sustained cabin pull: $1.5\text{--}2.0\,g$ toward the unit at $r \approx 1.5\text{--}2.0\,\text{m}$
 *       (physiological limits: blackout $8\text{--}9\,g$, redout $-3\,g$).
 *     - Exterior hull temperature: $100^\circ\text{F}\text{--}120^\circ\text{F}$ during transit.
 *     - Optical transition: $\sim 1\,\text{s}$ rainbow lensing & UV burst during ramp-up,
 *       followed by complete optical blackout until deactivation.
 *     - Ground ergosphere scoop: $\approx 1\,\text{ft}^3$ soil over $\approx 6\,\text{ft}^2$
 *       ($\sim 1\,\text{in}$ deep), with negative ergosphere front/back and positive center.
 *
 * ### 2. Exact Foundational General Relativity Equations
 *   - **Van Stockum (1937) / Tipler (1974) Rotating Cylinder:** Weyl-Papapetrou condition
 *     $F L + M^2 = r^2$, interior CTL boundary $a r > 1$, and supercritical exterior CTC
 *     threshold $a R > \frac{1}{2}$.
 *   - **Kerr-Newman Geometry (MTW / Wald / Hawking & Ellis / Pohlman `US20060073976A1`):**
 *     $\Delta(r) = r^2 - 2Mr + a^2 + Q^2$, super-extremal naked-singularity regime
 *     $a^2 + Q^2 > M^2$, and equatorial prograde/retrograde circular photon orbits $r_1, r_2$.
 *   - **Van Den Broeck / Alcubierre Screened Warp Envelope:** Smooth ellipsoidal shaping
 *     function $f(r_{\text{ell}}) \in [0, 1]$ confining curvature inside the $22\text{m}\times 6\text{m}\times 6\text{m}$
 *     bubble so the exterior ADM mass vanishes ($g_{00} \to -1$) without crushing the vehicle.
 *
 * ### 3. Calibrated 3+1 ADM Effective Phenomenological Metric
 *   - $ds^2 = -(\alpha^2 - \|\boldsymbol{\beta}\|^2) c^2 dt^2 + 2\beta_i c\,dt\,dx^i + \gamma_{ij} dx^i dx^j$
 *   - $g_{00}^{\text{eff}} = -(\alpha^2 - \|\boldsymbol{\beta}\|^2)$ reproduces the nested
 *     Negative ($g_{00} > 0$), Null ($g_{00} = 0$), and Positive ($g_{00} < 0$) Time Event
 *     Horizons while an analytically differentiated screened potential $\Phi_{\text{cabin}}(x,y,z)$
 *     reproduces the exact $1.5\text{--}2.0\,g$ cabin gradient $\partial g_{00}/\partial r = 2a/c^2$
 *     without IEEE-754 double-precision cancellation.
 *
 * @note Formal Lean 4 counterparts live in `formal/Umgebung/Unified/TitorC204.lean`.
 */

#pragma once

#include "umgebung/models/thinkers/ThinkerConstants.hpp"

#include <cmath>
#include <cstdint>

namespace umgebung::models::thinkers {

    /**
     * @namespace c204
     * @brief Verbatim physical, geometric, electrical, and physiological constants
     *        of the GE Army Model C204 Gravity Distortion Time Displacement Unit.
     */
    namespace c204 {
        /** @brief Standard Earth surface gravitational acceleration $g_0$ ($\text{m/s}^2$, exact SI). */
        inline constexpr double kStandardGravitySI = 9.80665;

        /** @brief Julian year length in hours ($365.25 \times 24 = 8766.0\,\text{hr/yr}$). */
        inline constexpr double kHoursPerJulianYear = 8766.0;

        /** @brief Seconds per hour ($3600.0\,\text{s/hr}$). */
        inline constexpr double kSecondsPerHour = 3600.0;

        /** @brief Unit dry mass in kilograms (`C204_Operators_Manual.pdf`: `225 kg`). */
        inline constexpr double kUnitDryMassKg = 225.0;

        /** @brief Maximum gross vehicle + unit + crew loading capacity (`2500 kg`). */
        inline constexpr double kMaxPayloadMassKg = 2500.0;

        /** @brief Unit physical housing height ($44\,\text{cm} = 0.44\,\text{m}$). */
        inline constexpr double kUnitHeightM = 0.44;

        /** @brief Unit physical housing width / dual-singularity lateral separation ($48\,\text{cm} = 0.48\,\text{m}$). */
        inline constexpr double kUnitWidthM = 0.48;

        /** @brief Unit physical housing length ($140\,\text{cm} = 1.40\,\text{m}$). */
        inline constexpr double kUnitLengthM = 1.40;

        /** @brief Centered loading envelope total length (`22 m`, semi-axis $a = 11\,\text{m}$). */
        inline constexpr double kEnvelopeLengthM = 22.0;

        /** @brief Centered loading envelope total width (`6 m`, semi-axis $b = 3\,\text{m}$). */
        inline constexpr double kEnvelopeWidthM = 6.0;

        /** @brief Centered loading envelope total height & vertical safe distance (`6 m`, semi-axis $c = 3\,\text{m}$). */
        inline constexpr double kEnvelopeHeightM = 6.0;

        /** @brief Figure 2 Item 7: Mass offset distance from dual singularities (`5 m`). */
        inline constexpr double kMassOffsetM = 5.0;

        /** @brief Figure 2 Item 8: Rear mass boundary distance (`8 m`). */
        inline constexpr double kRearMassDistanceM = 8.0;

        /** @brief Figure 2 Item 9: Forward mass boundary distance (`11 m`). */
        inline constexpr double kForwardMassDistanceM = 11.0;

        /** @brief Longitudinal envelope center shift $x_{\text{offset}} = (11 - 8)/2 = +1.5\,\text{m}$. */
        inline constexpr double kDefaultEnvelopeOffsetXM = 1.5;

        /** @brief Figure 2 Item 5: Rearward X-ray radiation venting cone arc (`170 deg`). */
        inline constexpr double kXRayVentConeDeg = 170.0;

        /** @brief Electrical charge range at `50%` power (`126,000 C`). */
        inline constexpr double kCharge50PctCoulombs = 126000.0;

        /** @brief Electrical charge range at `75%` power (`201,600 C`). */
        inline constexpr double kCharge75PctCoulombs = 201600.0;

        /** @brief Electrical charge range at `100%` power (`288,000 C`, backward CTC travel). */
        inline constexpr double kCharge100PctCoulombs = 288000.0;

        /**
         * @brief Directional energy reduction multiplier for forward-in-time travel
         *        relative to causality-reversing backward travel (`CompleteTitor.pdf`:
         *        "Yes, you can travel into the future and it takes less energy than going into the past").
         */
        inline constexpr double kForwardTravelEnergyFactor = 0.75;

        /** @brief Wave penetration rate at `50%` power (`3.4 yrs/hr`). */
        inline constexpr double kRate50PctYrsPerHr = 3.4;

        /** @brief Wave penetration rate at `75%` power (`6.2 yrs/hr`). */
        inline constexpr double kRate75PctYrsPerHr = 6.2;

        /** @brief Wave penetration rate at `100%` power (`10.0 yrs/hr`). */
        inline constexpr double kRate100PctYrsPerHr = 10.0;

        /** @brief Variable Gravity Lock (VGL) baseline polling interval (`240 s`). */
        inline constexpr double kVglIntervalSec = 240.0;

        /** @brief Variable Gravity Lock (VGL) total electrical charge budget (`54,000 C`). */
        inline constexpr double kVglTotalCoulombs = 54000.0;

        /** @brief Temporal accuracy break-away horizon (`60 yrs`). */
        inline constexpr double kBreakAwayHorizonYears = 60.0;

        /** @brief Nominal worldline divergence rate at max power (`1.0%` per `60 yrs` = `1/60 %/yr`). */
        inline constexpr double kNominalDivergencePctPerYear = 1.0 / 60.0;

        /** @brief Number of onboard Cesium atomic clocks (`Cesium 14-1` main, `Cesium 10-1` backup). */
        inline constexpr std::uint32_t kCesiumClockCount = 4U;

        /** @brief Number of parallel navigation computers linked via Borda voting. */
        inline constexpr std::uint32_t kNavigationComputerCount = 3U;

        /** @brief Minimum physiological cabin acceleration at 100% power (`1.5 G`). */
        inline constexpr double kCabinGForceMin = 1.5;

        /** @brief Nominal upper cabin acceleration at 100% power (`2.0 G`). */
        inline constexpr double kCabinGForceMax = 2.0;

        /** @brief Physiological positive G-force blackout threshold (`8.0 G`). */
        inline constexpr double kBlackoutThresholdG = 8.0;

        /** @brief Physiological negative G-force redout threshold (`-3.0 G`). */
        inline constexpr double kRedoutThresholdG = -3.0;

        /** @brief Ambient exterior pre-flight temperature ($72^\circ\text{F}$). */
        inline constexpr double kAmbientTempF = 72.0;

        /** @brief Average exterior temperature at 50% power ($100^\circ\text{F}$). */
        inline constexpr double kExteriorTemp50PctF = 100.0;

        /** @brief Average exterior temperature at 100% power ($120^\circ\text{F}$). */
        inline constexpr double kExteriorTemp100PctF = 120.0;

        /** @brief Duration of ramp-up rainbow optical lensing and UV burst (`1.0 s`). */
        inline constexpr double kUvBurstDurationSec = 1.0;

        /** @brief Typical ground soil scoop volume per departure/arrival ($\approx 1.0\,\text{ft}^3 = 0.0283168\,\text{m}^3$). */
        inline constexpr double kGroundScoopVolumeM3 = 0.0283168;

        /** @brief Typical ground soil scoop footprint area ($\approx 6.0\,\text{ft}^2 = 0.557418\,\text{m}^2$). */
        inline constexpr double kGroundScoopAreaM2 = 0.557418;
    } // namespace c204

    /**
     * @enum C204OperationalState
     * @brief Discrete operational state-machine phases of the GE C204 flight computer,
     *        mirroring the checklist sequence in `C204_Operators_Manual.pdf` and `CompleteTitor.pdf`.
     */
    enum class C204OperationalState : std::uint8_t {
        /** @brief Pre-flight stationary check (engine off, brake set, windows sealed, mass <= 2500 kg). */
        Standby = 0,
        /** @brief Initial Variable Gravity Lock (VGL) baseline gravity snapshot & Tipler alignment. */
        VglBaselineLock = 1,
        /** @brief Countdown alarm, field expansion, 1-second rainbow lensing + UV burst, G-force rise. */
        RampUp = 2,
        /** @brief Full temporal displacement under optical blackout at calibrated wave penetration rate. */
        TemporalTransit = 3,
        /** @brief Active 240-second VGL pulsed gravity sampling & Borda triple-computer verification. */
        VglPulseCheck = 4,
        /** @brief Arrival power step-down, horizon collapse, static discharge, daylight return. */
        ArrivalCooldown = 5,
        /** @brief Failsafe Automatic Origin Return triggered by VGL anomaly, mass overload, or manual abort. */
        OriginReturnAbort = 6
    };

    /**
     * @brief Converts `C204OperationalState` into a concise human-readable telemetry string.
     *
     * @param state Active state enumerator.
     * @return Null-terminated static label string.
     */
    [[nodiscard]] inline const char* C204OperationalStateToString(C204OperationalState state) noexcept {
        switch (state) {
            case C204OperationalState::Standby:           return "Standby (Pre-Check)";
            case C204OperationalState::VglBaselineLock:   return "VGL Baseline Lock";
            case C204OperationalState::RampUp:            return "Ramp-Up (Rainbow + UV Burst)";
            case C204OperationalState::TemporalTransit:   return "Temporal Transit (Blackout)";
            case C204OperationalState::VglPulseCheck:     return "VGL 240s Gravity Pulse Check";
            case C204OperationalState::ArrivalCooldown:   return "Arrival & Field Collapse";
            case C204OperationalState::OriginReturnAbort: return "AUTOMATIC ORIGIN RETURN (ABORT)";
        }
        return "Unknown";
    }

    /**
     * @struct C204MetricPointSample
     * @brief Complete point-wise evaluation of the C204 Effective Phenomenological
     *        3+1 ADM Metric at a 3D spatial point $(x, y, z)$ and time $t$.
     */
    struct C204MetricPointSample {
        /** @brief Dimensionless ellipsoidal radius $r_{\text{ell}}$ (`<= 1` inside `22m x 6m x 6m` envelope). */
        double ellipsoidalRadius{0.0};

        /** @brief Van Den Broeck / Alcubierre smooth spatial shaping function $f(r_{\text{ell}}) \in [0, 1]$. */
        double shapingFactorF{0.0};

        /** @brief 3+1 ADM lapse function $\alpha(x, y, z, t)$ driven by the damped offset Tipler sinusoid. */
        double lapseAlpha{1.0};

        /** @brief Magnitude of the 3+1 ADM frame-dragging shift vector $\|\boldsymbol{\beta}(x, y, z, t)\|$. */
        double shiftBetaMag{0.0};

        /** @brief Effective ADM time-time metric component $g_{00}^{\text{eff}} = -(\alpha^2 - \|\boldsymbol{\beta}\|^2)$. */
        double metricG00{-1.0};

        /** @brief Deviation from flat Minkowski spacetime $|1 + g_{00}^{\text{eff}}|$ (`0` exterior, `1` on Null Horizon, `>1` inside Negative Horizon). */
        double distortionMagnitude{0.0};

        /** @brief Local proper gravitational acceleration magnitude $a / g_0$ (in Earth G's, `1.5–2.0 G` at cabin). */
        double properAccelG{0.0};

        /** @brief Corresponding relativistic metric gradient magnitude $|\nabla g_{00}^{\text{local}}| = 2a/c^2$ ($\text{m}^{-1}$). */
        double metricGradientInvM{0.0};

        /** @brief True when $(x, y, z)$ lies inside the `170 deg` aft X-ray radiation venting cone. */
        bool inXRayVentCone{false};

        /**
         * @brief Active envelope horizon classification:
         *        `-1` = Negative Time Event Horizon ($g_{00} > +0.08$, backward CTC core),
         *        `0`  = Null Time Event Horizon / Cancellation Shell ($-0.10 \le g_{00} \le +0.08$),
         *        `+1` = Positive Time Field ($g_{00} < -0.10$ inside active `22m x 6m` envelope),
         *        `+2` = Flat Exterior Minkowski Space (outside Van Den Broeck envelope).
         */
        int horizonZone{2};
    };

    // =========================================================================
    // Section 1: Exact Verbatim Foundational Relativity Equations
    // =========================================================================

    /**
     * @brief Evaluates the Kerr-Newman radial horizon discriminant function
     *        $\Delta(r) = r^2 - 2 M r + a^2 + Q^2$ in geometric units ($G = c = 1$).
     *
     * @details
     * Cited verbatim from Wald, MTW, Hawking & Ellis, and Pohlman `US 2006/0073976 A1`.
     * Roots $r_\pm = M \pm \sqrt{M^2 - a^2 - Q^2}$ define the outer event horizon and
     * inner Cauchy horizon. When $a^2 + Q^2 > M^2$ (super-extremal regime), $\Delta(r) > 0$
     * everywhere, exposing the naked ring singularity and chronology-violating region.
     *
     * @param r Radial Boyer-Lindquist coordinate $r$ (geometric units).
     * @param massM Mass parameter $M > 0$.
     * @param spinA Angular momentum per unit mass $a = J/M$.
     * @param chargeQ Electric charge parameter $Q$.
     * @return Value of $\Delta(r)$.
     * @note Formalized in `formal/Umgebung/Unified/TitorC204.lean` (`kerrNewman_extremal_double_root`).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeKerrNewmanDelta(
        double r,
        double massM,
        double spinA,
        double chargeQ
    ) noexcept {
        return (r * r) - (2.0 * massM * r) + (spinA * spinA) + (chargeQ * chargeQ);
    }

    /**
     * @brief Returns `true` when the Kerr-Newman singularity satisfies the
     *        super-extremal naked-singularity condition $a^2 + Q^2 > M^2$.
     *
     * @param massM Mass parameter $M$.
     * @param spinA Spin parameter $a$.
     * @param chargeQ Electric charge parameter $Q$.
     * @return `true` iff $a^2 + Q^2 > M^2$.
     */
    UMGEBUNG_HD [[nodiscard]] inline bool IsKerrNewmanSuperExtremal(
        double massM,
        double spinA,
        double chargeQ
    ) noexcept {
        return ((spinA * spinA) + (chargeQ * chargeQ)) > (massM * massM);
    }

    /**
     * @brief Computes the equatorial prograde circular photon orbit radius $r_1$
     *        around a rotating Kerr black hole (Pohlman `US 2006/0073976 A1`):
     *        $$r_1 = 2 M \left[1 + \cos\!\left(\frac{2}{3}\arccos\!\left(-\frac{|a|}{M}\right)\right)\right]$$
     *
     * @details Satisfies $M \le r_1 \le 3M$, reducing to the Schwarzschild photon
     *          sphere $r_1 = 3M$ when $a = 0$.
     *
     * @param massM Black hole mass parameter $M > 0$.
     * @param spinA Spin parameter $a$ (clamped internally to $|a|/M \le 1$).
     * @return Prograde equatorial photon orbit radius $r_1$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeKerrProgradePhotonRadius(
        double massM,
        double spinA
    ) noexcept {
        if (massM <= 0.0) {
            return 0.0;
        }
        double chi = fabs(spinA) / massM;
        if (chi > 1.0) {
            chi = 1.0;
        }
        return 2.0 * massM * (1.0 + cos((2.0 / 3.0) * acos(-chi)));
    }

    /**
     * @brief Computes the equatorial retrograde circular photon orbit radius $r_2$
     *        around a rotating Kerr black hole (Pohlman `US 2006/0073976 A1`):
     *        $$r_2 = 2 M \left[1 + \cos\!\left(\frac{2}{3}\arccos\!\left(+\frac{|a|}{M}\right)\right)\right]$$
     *
     * @details Satisfies $3M \le r_2 \le 4M$, reducing to $r_2 = 3M$ when $a = 0$.
     *          Differential electron bombardment exploits the $r_1 < r_2$ Lense-Thirring
     *          asymmetry to drive the offset Tipler sinusoid.
     *
     * @param massM Black hole mass parameter $M > 0$.
     * @param spinA Spin parameter $a$ (clamped internally to $|a|/M \le 1$).
     * @return Retrograde equatorial photon orbit radius $r_2$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeKerrRetrogradePhotonRadius(
        double massM,
        double spinA
    ) noexcept {
        if (massM <= 0.0) {
            return 0.0;
        }
        double chi = fabs(spinA) / massM;
        if (chi > 1.0) {
            chi = 1.0;
        }
        return 2.0 * massM * (1.0 + cos((2.0 / 3.0) * acos(chi)));
    }

    /**
     * @brief Evaluates the van Stockum (1937) / Tipler (1974) interior rotating
     *        cylinder azimuthal metric component $L(r) = r^2 (1 - a^2 r^2)$.
     *
     * @details Together with $F = 1$ and $M = a r^2$, satisfies the exact Weyl-Papapetrou
     *          coordinate condition $F L + M^2 = r^2$ (proven in `TitorC204.lean`).
     *          Becomes negative for $r > 1/a$, creating closed timelike lines (CTLs).
     *
     * @param r Radial distance from cylinder axis $r \ge 0$.
     * @param angularVelA Cylinder angular velocity parameter $a$.
     * @return Interior azimuthal metric component $g_{\phi\phi} = L(r)$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVanStockumInteriorL(
        double r,
        double angularVelA
    ) noexcept {
        const double ar = angularVelA * r;
        return (r * r) * (1.0 - (ar * ar));
    }

    /**
     * @brief Evaluates Frank J. Tipler's (1974, Phys. Rev. D 9, Eq. 3c) supercritical
     *        exterior rotating cylinder azimuthal metric component $L(r)$ for $aR > \frac{1}{2}$:
     *        $$L(r) = \frac{R r \sin(3\beta + \gamma)}{2 \sin(2\beta)\cos\beta}, \quad
     *          \beta = \arctan\sqrt{4 a^2 R^2 - 1}, \quad
     *          \gamma = \sqrt{4 a^2 R^2 - 1}\,\ln(r/R)$$
     *
     * @details Because $\sin(3\beta + \gamma)$ oscillates through negative values as $r/R$
     *          increases, $g_{\phi\phi} = L(r) < 0$ arises outside the cylinder with no event
     *          horizon, exposing global closed timelike curves whenever $aR > \frac{1}{2}$.
     *
     * @param r Exterior radial coordinate $r \ge R > 0$.
     * @param cylinderRadiusR Cylinder boundary radius $R > 0$.
     * @param aR Dimensionless cylinder rotation speed $aR$ (must exceed `0.5` for supercritical regime).
     * @return Exterior azimuthal metric component $L(r)$ (or `0.0` if inputs are out of supercritical range).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeTiplerSupercriticalExteriorL(
        double r,
        double cylinderRadiusR,
        double aR
    ) noexcept {
        if (r <= 0.0 || cylinderRadiusR <= 0.0 || aR <= 0.5) {
            return 0.0;
        }
        const double disc = sqrt((4.0 * aR * aR) - 1.0);
        const double beta = atan(disc);
        const double gamma = disc * log(r / cylinderRadiusR);
        const double denom = 2.0 * sin(2.0 * beta) * cos(beta);
        if (fabs(denom) < 1.0e-15) {
            return 0.0;
        }
        return (cylinderRadiusR * r * sin((3.0 * beta) + gamma)) / denom;
    }

    // =========================================================================
    // Section 2: Verbatim C204 Operational Transfer Functions & Telemetry
    // =========================================================================

    /**
     * @brief Computes the C204 electron-injection charge requirement $Q(P)$ in Coulombs
     *        as a function of power setting $P \in [0, 100]\%$.
     *
     * @details Reproduces the three verbatim manual calibration points to exact
     *          machine precision:
     *            - $P = 50\%  \implies 126{,}000\,\text{C}$
     *            - $P = 75\%  \implies 201{,}600\,\text{C}$
     *            - $P = 100\% \implies 288{,}000\,\text{C}$
     *          with smooth monotonic piecewise interpolation across intermediate settings.
     *
     * @param powerPercent Power setting in percent $[0, 100]$.
     * @return Required electron injection charge in Coulombs.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204ChargeForPower(double powerPercent) noexcept {
        if (powerPercent <= 0.0) {
            return 0.0;
        }
        if (powerPercent <= 50.0) {
            // Smooth quadratic-linear ramp reaching exactly 126,000 C at 50%.
            const double u = powerPercent / 50.0;
            return c204::kCharge50PctCoulombs * u;
        }
        if (powerPercent <= 75.0) {
            const double u = (powerPercent - 50.0) / 25.0;
            return c204::kCharge50PctCoulombs +
                   u * (c204::kCharge75PctCoulombs - c204::kCharge50PctCoulombs);
        }
        if (powerPercent <= 100.0) {
            const double u = (powerPercent - 75.0) / 25.0;
            return c204::kCharge75PctCoulombs +
                   u * (c204::kCharge100PctCoulombs - c204::kCharge75PctCoulombs);
        }
        // Extrapolate safely above 100% if explicitly requested by an experiment.
        const double u = (powerPercent - 75.0) / 25.0;
        return c204::kCharge75PctCoulombs +
               u * (c204::kCharge100PctCoulombs - c204::kCharge75PctCoulombs);
    }

    /**
     * @brief Computes directional electron-injection charge $Q(P, \text{dir})$ in Coulombs,
     *        accounting for John Titor's verbatim energy asymmetry (`CompleteTitor.pdf`):
     *        forward temporal displacement aligns with natural gravitational time dilation
     *        and requires `kForwardTravelEnergyFactor` (`75%`) of the charge required to
     *        drive causality-reversing backward CTC displacement.
     *
     * @param powerPercent Active power setting in percent $[0, 100]$.
     * @param isForwardTravel `true` for future displacement ($\Delta t > 0$), `false` for past ($\Delta t \le 0$).
     * @return Required directional electron injection charge in Coulombs.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204DirectionalCharge(
        double powerPercent,
        bool isForwardTravel
    ) noexcept {
        const double baseCharge = ComputeC204ChargeForPower(powerPercent);
        return isForwardTravel ? (baseCharge * c204::kForwardTravelEnergyFactor) : baseCharge;
    }

    /**
     * @brief Computes the C204 wave penetration rate in years per pilot hour
     *        as a function of power setting $P \in [0, 100]\%$.
     *
     * @details Reproduces the three verbatim manual rates identically:
     *            - $P = 50\%  \implies 3.4\,\text{yrs/hr}$
     *            - $P = 75\%  \implies 6.2\,\text{yrs/hr}$
     *            - $P = 100\% \implies 10.0\,\text{yrs/hr}$
     *
     * @param powerPercent Power setting in percent $[0, 100]$.
     * @return Wave penetration rate in external years per pilot proper hour.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204PenetrationRateYrsPerHr(double powerPercent) noexcept {
        if (powerPercent <= 0.0) {
            return 0.0;
        }
        if (powerPercent <= 50.0) {
            const double u = powerPercent / 50.0;
            return c204::kRate50PctYrsPerHr * u;
        }
        if (powerPercent <= 75.0) {
            const double u = (powerPercent - 50.0) / 25.0;
            return c204::kRate50PctYrsPerHr +
                   u * (c204::kRate75PctYrsPerHr - c204::kRate50PctYrsPerHr);
        }
        const double u = (powerPercent - 75.0) / 25.0;
        return c204::kRate75PctYrsPerHr +
               u * (c204::kRate100PctYrsPerHr - c204::kRate75PctYrsPerHr);
    }

    /**
     * @brief Converts a wave penetration rate (years/hour) into the dimensionless
     *        temporal coordinate translation velocity $v_t = |dt_{\text{external}} / d\tau_{\text{pilot}}|$
     *        (external coordinate seconds per pilot proper second).
     *
     * @details At $100\%$ power ($10\,\text{yrs/hr}$):
     *          $$v_t(100\%) = 10 \times 365.25 \times 24 = 87{,}660\,\text{s/s}.$$
     *
     * @param penetrationRateYrsPerHr Wave penetration rate in $\text{yrs/hr}$.
     * @return Temporal coordinate velocity $v_t$ in $\text{s/s}$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204CoordinateTimeVelocity(
        double penetrationRateYrsPerHr
    ) noexcept {
        return (penetrationRateYrsPerHr > 0.0)
            ? (penetrationRateYrsPerHr * c204::kHoursPerJulianYear)
            : 0.0;
    }

    /**
     * @brief Computes the empirical charge-to-temporal-velocity power-law fit
     *        $v_t(Q) = k \cdot Q^{1.25}$ calibrated to `288,000 C -> 87,660 s/s`
     *        (from the John Titor notebook Turn 10 phenomenological calibration).
     *
     * @param chargeCoulombs Injected charge $Q \ge 0$ in Coulombs.
     * @return Calibrated coordinate time velocity $v_t(Q)$ in $\text{s/s}$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204PowerLawTimeVelocity(
        double chargeCoulombs
    ) noexcept {
        if (chargeCoulombs <= 0.0) {
            return 0.0;
        }
        // Calibrated so Q = 288,000 C yields exactly 87,660 s/s (10 yrs/hr).
        const double qNorm = chargeCoulombs / c204::kCharge100PctCoulombs;
        return (c204::kRate100PctYrsPerHr * c204::kHoursPerJulianYear) * pow(qNorm, 1.25);
    }

    /**
     * @brief Estimates cumulative worldline divergence percentage as a function of
     *        elapsed temporal displacement, VGL check count, and Borda variance.
     *
     * @details
     * Implements Titor's stated divergence rules verbatim:
     *   - Nominal baseline: $1.0\%$ per $60\,\text{years}$ at maximum power
     *     ($0.016667\%/\text{yr}$).
     *   - Break-away regime ($|\Delta t| > 60\,\text{yr}$): quadratic chaotic-attractor
     *     drift penalty reflecting rapidly degrading VGL confidence beyond $60\,\text{years}$.
     *   - VGL synchronization jitter: adds small Borda error term $\sigma_{\text{Borda}}$
     *     per VGL check (reproducing $\sim 2.5\%$ divergence across the $61$-year $2036 \to 1975$
     *     transit when VGL checks introduce realistic terrestrial gravity variance).
     *
     * @param elapsedYears Absolute displaced temporal span $|\Delta t|$ in years.
     * @param vglCheckCount Number of completed 240-second VGL lock cycles.
     * @param bordaVariance Accumulated Borda sensor consensus variance ($\ge 0$).
     * @return Estimated worldline divergence in percent ($\ge 0$).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204WorldlineDivergencePct(
        double elapsedYears,
        std::uint64_t vglCheckCount,
        double bordaVariance
    ) noexcept {
        const double absYears = fabs(elapsedYears);
        double baseDiv = c204::kNominalDivergencePctPerYear * absYears;
        if (absYears > c204::kBreakAwayHorizonYears) {
            const double excess = (absYears - c204::kBreakAwayHorizonYears) / c204::kBreakAwayHorizonYears;
            // Rapidly growing chaotic-attractor break-away beyond 60 years.
            baseDiv += 4.0 * excess * excess;
        }
        const double vglDrift = 0.012 * static_cast<double>(vglCheckCount);
        const double bordaTerm = (bordaVariance > 0.0) ? (10.0 * bordaVariance) : 0.0;
        return baseDiv + vglDrift + bordaTerm;
    }

    /**
     * @brief Computes the exterior vehicle hull temperature in degrees Fahrenheit
     *        as a function of power setting and ramp-up envelope $u_{\text{ramp}} \in [0, 1]$.
     *
     * @details Transitions smoothly from ambient ($72^\circ\text{F}$) to
     *          $100^\circ\text{F}$ at $50\%$ power and $120^\circ\text{F}$ at $100\%$ power
     *          (`CompleteTitor.pdf`: "100 to 120 degrees is average depending on power setting").
     *
     * @param powerPercent Active power setting in percent $[0, 100]$.
     * @param rampFraction Instantaneous field ramp factor $u_{\text{ramp}} \in [0, 1]$.
     * @return Exterior hull temperature in $^\circ\text{F}$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204ExteriorTempF(
        double powerPercent,
        double rampFraction
    ) noexcept {
        const double pClamped = (powerPercent < 0.0) ? 0.0 : ((powerPercent > 100.0) ? 100.0 : powerPercent);
        const double uClamped = (rampFraction < 0.0) ? 0.0 : ((rampFraction > 1.0) ? 1.0 : rampFraction);
        double steadyTempF = c204::kAmbientTempF;
        if (pClamped <= 50.0) {
            const double f = pClamped / 50.0;
            steadyTempF = c204::kAmbientTempF + f * (c204::kExteriorTemp50PctF - c204::kAmbientTempF);
        } else {
            const double f = (pClamped - 50.0) / 50.0;
            steadyTempF = c204::kExteriorTemp50PctF + f * (c204::kExteriorTemp100PctF - c204::kExteriorTemp50PctF);
        }
        return c204::kAmbientTempF + uClamped * (steadyTempF - c204::kAmbientTempF);
    }

    /**
     * @brief Performs fault-tolerant **Borda error-correcting consensus voting** across
     *        the C204's 3 parallel navigation computers (capable of full operation with
     *        1 unit disabled per `C204_Operators_Manual.pdf`).
     *
     * @details
     * When all 3 computers are online, ranks pairwise agreements to down-weight any
     * single outlier sensor (Borda median-consensus) and reports the maximum residual
     * among active computers. When 1 computer is disabled, averages the remaining 2
     * computers cleanly (`(g + g) / 2 = g`, proven in `TitorC204.lean`).
     *
     * @param r0 Gravity sensor reading from Computer #1.
     * @param r1 Gravity sensor reading from Computer #2.
     * @param r2 Gravity sensor reading from Computer #3.
     * @param active0 True if Computer #1 is online.
     * @param active1 True if Computer #2 is online.
     * @param active2 True if Computer #3 is online.
     * @param outConsensus Output voted consensus reading.
     * @param outSpread Output disagreement spread across active computers.
     * @return `true` if at least 2 navigation computers are operational (manual redundancy requirement).
     */
    UMGEBUNG_HD [[nodiscard]] inline bool EvaluateC204BordaConsensus(
        double r0,
        double r1,
        double r2,
        bool active0,
        bool active1,
        bool active2,
        double& outConsensus,
        double& outSpread
    ) noexcept {
        const int onlineCount = (active0 ? 1 : 0) + (active1 ? 1 : 0) + (active2 ? 1 : 0);
        if (onlineCount < 2) {
            outConsensus = 0.0;
            outSpread = 1.0e9;
            return false;
        }
        if (onlineCount == 2) {
            double a = 0.0;
            double b = 0.0;
            if (!active0)      { a = r1; b = r2; }
            else if (!active1) { a = r0; b = r2; }
            else               { a = r0; b = r1; }
            outConsensus = 0.5 * (a + b);
            outSpread = fabs(a - b);
            return true;
        }
        // All 3 computers online: compute pairwise discrepancies and apply Borda rank-weighting
        // so an isolated single-channel glitch cannot corrupt the VGL consensus lock.
        const double d01 = fabs(r0 - r1);
        const double d02 = fabs(r0 - r2);
        const double d12 = fabs(r1 - r2);
        const double e0 = d01 + d02;
        const double e1 = d01 + d12;
        const double e2 = d02 + d12;

        // If all three closely agree, return exact arithmetic mean (matching Lean unanimity).
        const double maxPair = (d01 > d02) ? ((d01 > d12) ? d01 : d12) : ((d02 > d12) ? d02 : d12);
        if (maxPair <= 1.0e-9) {
            outConsensus = (r0 + r1 + r2) / 3.0;
            outSpread = maxPair;
            return true;
        }
        // Otherwise exclude the highest-error channel (single-fault rejection) and average the agreeing pair.
        if (e0 >= e1 && e0 >= e2) {
            outConsensus = 0.5 * (r1 + r2);
            outSpread = d12;
        } else if (e1 >= e0 && e1 >= e2) {
            outConsensus = 0.5 * (r0 + r2);
            outSpread = d02;
        } else {
            outConsensus = 0.5 * (r0 + r1);
            outSpread = d01;
        }
        return true;
    }

    // =========================================================================
    // Section 3: Calibrated 3+1 ADM Effective Phenomenological Metric
    // =========================================================================

    /**
     * @brief Computes the dimensionless ellipsoidal coordinate $r_{\text{ell}}(x, y, z)$
     *        for the C204 $22\,\text{m} \times 6\,\text{m} \times 6\,\text{m}$ loading envelope:
     *        $$r_{\text{ell}} = \sqrt{\left(\frac{x - x_{\text{offset}}}{a}\right)^2 + \left(\frac{y}{b}\right)^2 + \left(\frac{z}{c}\right)^2}$$
     *
     * @param x Longitudinal coordinate $x$ in meters (`+x` = forward toward front bumper/cab).
     * @param y Lateral coordinate $y$ in meters.
     * @param z Vertical coordinate $z$ in meters.
     * @param offsetXM Longitudinal envelope center offset $x_{\text{offset}}$ (default `+1.5 m`).
     * @param semiAxisA Semi-major length $a$ (default `11.0 m`, total `22 m`).
     * @param semiAxisB Semi-minor width $b$ (default `3.0 m`, total `6 m`).
     * @param semiAxisC Semi-minor height $c$ (default `3.0 m`, total `6 m`).
     * @return Dimensionless ellipsoidal radius $r_{\text{ell}} \ge 0$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeC204EllipsoidalRadius(
        double x,
        double y,
        double z,
        double offsetXM,
        double semiAxisA,
        double semiAxisB,
        double semiAxisC
    ) noexcept {
        const double ux = (semiAxisA > 0.0) ? ((x - offsetXM) / semiAxisA) : 0.0;
        const double uy = (semiAxisB > 0.0) ? (y / semiAxisB) : 0.0;
        const double uz = (semiAxisC > 0.0) ? (z / semiAxisC) : 0.0;
        return sqrt((ux * ux) + (uy * uy) + (uz * uz));
    }

    /**
     * @brief Evaluates the smooth Van Den Broeck / Alcubierre warp-bubble spatial
     *        shaping function $f(r_{\text{ell}}) \in [0, 1]$:
     *        $$f(r_{\text{ell}}) = \frac{\tanh(\sigma_w(1 + r_{\text{ell}})) + \tanh(\sigma_w(1 - r_{\text{ell}}))}{2 \tanh(\sigma_w)}$$
     *
     * @details Satisfies $f(0) = 1$ identically (formally proven in `TitorC204.lean`),
     *          transitions steeply across the $22\,\text{m}\times 6\,\text{m}\times 6\,\text{m}$
     *          envelope wall ($r_{\text{ell}} = 1$), and vanishes exponentially for $r_{\text{ell}} > 1$
     *          so the exterior ADM mass seen by the Earth and truck suspension is zero.
     *
     * @param rEll Dimensionless ellipsoidal radius $r_{\text{ell}} \ge 0$.
     * @param wallSteepness Wall transition steepness $\sigma_w > 0$ (default `4.0`).
     * @return Smooth spatial screening factor $f \in [0, 1]$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVanDenBroeckShapingFactor(
        double rEll,
        double wallSteepness
    ) noexcept {
        if (wallSteepness <= 0.0) {
            return (rEll <= 1.0) ? 1.0 : 0.0;
        }
        const double denom = 2.0 * tanh(wallSteepness);
        if (fabs(denom) < 1.0e-15) {
            return 0.0;
        }
        const double val = (tanh(wallSteepness * (1.0 + rEll)) + tanh(wallSteepness * (1.0 - rEll))) / denom;
        return (val < 0.0) ? 0.0 : ((val > 1.0) ? 1.0 : val);
    }

    /**
     * @brief Tests whether a point $(x, y)$ lies inside the C204's aft $170^\circ$
     *        X-ray radiation venting cone (Figure 2 Item 5, `C204_Operators_Manual.pdf`).
     *
     * @details Radiation is vented rearward (`-x` direction) across a total arc of
     *          `coneTotalDeg` (default `170 deg`, i.e., $\pm 85^\circ$ from the `-x` axis)
     *          behind the dual-singularity housing.
     *
     * @param x Longitudinal coordinate $x$ in meters.
     * @param y Lateral coordinate $y$ in meters.
     * @param singCenterX Longitudinal center of the dual singularities (default `0.0 m`).
     * @param coneTotalDeg Full angular opening of the aft venting cone in degrees (`170.0`).
     * @return `true` if $(x, y)$ lies within the rearward X-ray hazard cone.
     */
    UMGEBUNG_HD [[nodiscard]] inline bool IsPointInC204XRayVentCone(
        double x,
        double y,
        double singCenterX,
        double coneTotalDeg
    ) noexcept {
        const double dx = x - singCenterX;
        if (dx >= 0.0) {
            return false;
        }
        const double halfAngleRad = 0.5 * coneTotalDeg * (constants::kPi / 180.0);
        const double angleFromAft = atan2(fabs(y), -dx);
        return angleFromAft <= halfAngleRad;
    }

    /**
     * @brief Evaluates the complete calibrated **C204 Effective Phenomenological Metric**
     *        $g_{\mu\nu}^{\text{eff}}(t, x, y, z)$ on CPU or GPU (`__host__ __device__`).
     *
     * @details
     * Combines all boundary constraints derived from the John Titor notebook (Turns 8–10),
     * `C204_Operators_Manual.pdf` Figure 2, Pohlman `US 2006/0073976 A1`, and Tipler (1974):
     *   1. **Van Den Broeck Ellipsoidal Screening:** $f(r_{\text{ell}})$ confines the distortion
     *      cleanly inside the $22\,\text{m} \times 6\,\text{m} \times 6\,\text{m}$ offset envelope
     *      ($x_{\text{offset}} = +1.5\,\text{m}$).
     *   2. **Dual-Singularity Interference Distance:** Combines the Drive (`+y = +d/2`) and
     *      Steering (`-y = -d/2`) micro-singularities separated by $d = 0.48\,\text{m}$ with
     *      quadrupole $\cos(2\theta)$ asymmetry matching the front/rear negative ergosphere vs.
     *      central positive ergosphere ground footprint.
     *   3. **Damped Tipler Sinusoid Lapse $\alpha(x, y, z, t)$ & Shift $\beta(x, y, z, t)$:**
     *      Produces nested **Negative Time** ($g_{00}^{\text{eff}} > 0$), **Null Time**
     *      ($g_{00}^{\text{eff}} = 0$), and **Positive Time** ($g_{00}^{\text{eff}} < 0$)
     *      event horizons scaling smoothly with electron injection charge $Q$ and ramp envelope $u_{\text{ramp}}$.
     *   4. **FP64-Safe Cabin Proper Acceleration $a(x,y,z)/g_0$:**
     *      Analytically evaluates $\|\nabla \Phi_{\text{cabin}}\|$ from a Van Den Broeck screened
     *      radial potential calibrated to produce **$1.85\,g$** (strictly within Titor's
     *      $1.5\text{--}2.0\,g$ window) at the pilot cabin ($r = 1.8\,\text{m}$) at $100\%$ power,
     *      while reporting the exact relativistic metric derivative
     *      $|\nabla g_{00}^{\text{local}}| = 2a / c^2 \approx 4.0 \times 10^{-16}\,\text{m}^{-1}$
     *      without double-precision catastrophic cancellation.
     *
     * @param x Longitudinal coordinate (m).
     * @param y Lateral coordinate (m).
     * @param z Vertical coordinate (m).
     * @param simTime Pilot proper time $t$ (s).
     * @param powerFraction Normalized power fraction $P/100 \in [0, 1]$.
     * @param rampFraction Smooth operational ramp envelope $u_{\text{ramp}} \in [0, 1]$.
     * @param offsetXM Envelope longitudinal offset $x_{\text{offset}}$ (m).
     * @param semiAxisA Envelope semi-major length $a$ (m).
     * @param semiAxisB Envelope semi-minor width $b$ (m).
     * @param semiAxisC Envelope semi-minor height $c$ (m).
     * @param singularitySpacingM Dual-singularity separation $d_{\text{sing}}$ (m, default `0.48`).
     * @param cabinProbeDistM Reference pilot cabin distance from singularities (m, default `1.8`).
     * @param tiplerWavelengthM Spatial wavelength $\lambda_{\text{Tipler}}$ of concentric horizons (m).
     * @param dampingScaleM Radial decay length $\sigma_{\text{damp}}$ (m).
     * @param wallSteepness Van Den Broeck wall steepness $\sigma_{\text{wall}}$.
     * @param xrayVentConeDeg Aft X-ray venting cone angle (deg, default `170.0`).
     * @param tiplerAR Supercritical Tipler rotation parameter $aR$ (default `0.75`).
     * @param isForwardTravel `true` when traveling forward in time ($\Delta t > 0$,
     *                        prograde Positive Time dilation mode requiring less energy
     *                        with no causality-reversing Negative Time CTC core);
     *                        `false` when traveling backward ($\Delta t \le 0$, retrograde
     *                        supercritical Negative Time CTC mode).
     * @param vglPulseEnvelope Smooth $[0, 1]$ envelope active strictly during a 240-second
     *                         VGL gravity re-synchronization pulse (`0.0` during steady transit).
     * @return Fully populated `C204MetricPointSample`.
     */
    UMGEBUNG_HD [[nodiscard]] inline C204MetricPointSample EvaluateC204EffectiveMetricPoint(
        double x,
        double y,
        double z,
        double simTime,
        double powerFraction,
        double rampFraction,
        double offsetXM,
        double semiAxisA,
        double semiAxisB,
        double semiAxisC,
        double singularitySpacingM,
        double cabinProbeDistM,
        double tiplerWavelengthM,
        double dampingScaleM,
        double wallSteepness,
        double xrayVentConeDeg,
        double tiplerAR,
        bool isForwardTravel = false,
        double vglPulseEnvelope = 0.0
    ) noexcept {
        C204MetricPointSample sample{};

        const double pClamped = (powerFraction < 0.0) ? 0.0 : ((powerFraction > 1.2) ? 1.2 : powerFraction);
        const double uClamped = (rampFraction < 0.0) ? 0.0 : ((rampFraction > 1.0) ? 1.0 : rampFraction);
        // Forward travel requires less electrical energy (~75% charge -> ~86.6% field/gravity drive),
        // keeping cabin G-force comfortably inside 1.5-2.0 G while reflecting lower forward energy demand.
        const double effectivePower = isForwardTravel
            ? (pClamped * c204::kForwardTravelEnergyFactor)
            : pClamped;
        const double cabinDriveScale = uClamped * pClamped * (isForwardTravel ? 0.87 : 1.0);

        // 1. Van Den Broeck / Alcubierre ellipsoidal envelope radius and smooth screening factor.
        //    Using `envWeight = fEnv^2` keeps full field intensity across the entire 22 m x 6 m x 6 m
        //    oval interior while guaranteeing rapid asymptotic screening to flat Minkowski spacetime
        //    (g_00 -> -1) immediately outside the envelope boundary.
        const double rEll = ComputeC204EllipsoidalRadius(x, y, z, offsetXM, semiAxisA, semiAxisB, semiAxisC);
        const double fEnv = ComputeVanDenBroeckShapingFactor(rEll, wallSteepness);
        const double envWeight = fEnv * fEnv;
        sample.ellipsoidalRadius = rEll;
        sample.shapingFactorF = fEnv;

        // 2. Dual positive top-spin micro-singularities positioned laterally at (0, +/- d/2, 0).
        const double halfSep = 0.5 * ((singularitySpacingM > 0.0) ? singularitySpacingM : c204::kUnitWidthM);
        const double r1 = sqrt((x * x) + ((y - halfSep) * (y - halfSep)) + (z * z));
        const double r2 = sqrt((x * x) + ((y + halfSep) * (y + halfSep)) + (z * z));
        const double rCenter = sqrt((x * x) + (y * y) + (z * z));
        // Ellipsoidal azimuth around the offset envelope center + near-field dual-singularity accent.
        const double thetaEnv = atan2(y / ((semiAxisB > 0.0) ? semiAxisB : 3.0),
                                      (x - offsetXM) / ((semiAxisA > 0.0) ? semiAxisA : 11.0));
        // Quadrupole Y_20/Y_22 ergosphere footprint polarity (`CompleteTitor.pdf`):
        //   - Backward (negative ergosphere): enhanced front & rear lobes (+cos(2*theta)).
        //   - Forward  (positive ergosphere): elongated near the center of mass (-cos(2*theta)).
        const double quadSign = isForwardTravel ? -1.0 : 1.0;
        const double quadMod = 1.0 + (quadSign * 0.14 * cos(2.0 * thetaEnv));
        const double rSingPair = 0.5 * (r1 + r2);
        const double singCoreAccent = 0.18 * exp(-0.5 * (rSingPair / 1.4) * (rSingPair / 1.4));

        // 3. Quasi-stationary ellipsoidal offset Tipler sinusoid lapse function alpha(x, y, z, t):
        //    Parameterized over the full 22 m x 6 m x 6 m ellipsoidal coordinate `rEll` so the
        //    nested Negative / Null / Positive Time Event Horizons encompass the entire oval
        //    envelope (matching Figure 2 of `C204_Operators_Manual.pdf`). Holds spatially locked
        //    equilibrium during transit (GW & X-ray transit at c ~ 37 ns) with only sub-percent
        //    power-bus micro-ripple (~0.4%) and 240-second VGL gravity re-lock pulses.
        const double waveRatio = (tiplerWavelengthM > 1.0e-3) ? (3.84 / tiplerWavelengthM) : 1.0;
        const double sinusoidPhase = 1.5 * constants::kPi * rEll * waveRatio;
        const double ovalRadialProfile = (0.58 + 0.42 * cos(sinusoidPhase)) * exp(-0.22 * rEll) + singCoreAccent;
        const double microRipple = 1.0 + 0.004 * sin(2.0 * constants::kPi * 1.7 * simTime);
        const double vglClamp = (vglPulseEnvelope < 0.0) ? 0.0 : ((vglPulseEnvelope > 1.0) ? 1.0 : vglPulseEnvelope);
        const double vglTrimFactor = 1.0 + (0.06 * vglClamp);

        // Charge-calibrated lapse modulation amplitude A(Q): scales with (Q_eff / Q_100)^1.25.
        const double chargeNorm = pow((effectivePower > 0.0) ? effectivePower : 0.0, 1.25);
        const double lapseAmp = 0.68 * chargeNorm * quadMod * microRipple * vglTrimFactor;
        const double alpha = 1.0 - (uClamped * envWeight * lapseAmp * ovalRadialProfile);

        // 4. Ellipsoidal Kerr frame-dragging shift magnitude ||beta|| spanning the 22 m x 6 m oval:
        //    - Backward (Delta t <= 0): supercritical retrograde Tipler/Kerr dragging (||beta|| > |alpha|)
        //      opens an oval Negative Time Event Horizon (g_00 > 0) nested cleanly inside the oval
        //      Null Time Horizon (g_00 = 0) and outer Positive Time Field filling the full 22 m x 6 m envelope.
        //    - Forward  (Delta t > 0): prograde positive-time dilation stays strictly sub-critical
        //      (||beta|| <= 0.94 |alpha|, so g_00 <= 0 everywhere; proven in `TitorC204.lean`),
        //      energizing the entire 22 m x 6 m oval as a Positive Time Field + inner Null cancellation shell.
        const double superCritBoost = (tiplerAR > 0.5) ? (1.0 + 0.65 * (tiplerAR - 0.5)) : 0.5;
        const double dampScaleNorm = (dampingScaleM > 1.0e-3) ? (dampingScaleM / 6.5) : 1.0;
        const double shiftSigmaEll = 0.54 * dampScaleNorm;
        const double shiftProfile = exp(-0.5 * (rEll / shiftSigmaEll) * (rEll / shiftSigmaEll));
        double betaMag = 0.0;
        if (isForwardTravel) {
            const double rawBeta = uClamped * envWeight * (0.92 * chargeNorm) * quadMod * shiftProfile;
            const double subCritCeiling = 0.94 * fabs(alpha);
            betaMag = (rawBeta < subCritCeiling) ? rawBeta : subCritCeiling;
        } else {
            betaMag = uClamped * envWeight * (1.22 * chargeNorm * superCritBoost) * quadMod * shiftProfile;
        }

        // 5. Effective 3+1 ADM time-time metric component g_00^eff = -(alpha^2 - ||beta||^2).
        const double g00 = -((alpha * alpha) - (betaMag * betaMag));

        sample.lapseAlpha = alpha;
        sample.shiftBetaMag = betaMag;
        sample.metricG00 = g00;
        sample.distortionMagnitude = fabs(1.0 + g00);

        // Classify active 22m x 6m envelope regions vs. flat exterior Minkowski spacetime.
        const bool inActiveEnvelope = (fEnv > 0.10) && (sample.distortionMagnitude > 0.04);
        if (!inActiveEnvelope) {
            sample.horizonZone = 2;  // Flat exterior Minkowski space (g_00 ~ -1)
        } else if (g00 > 0.08) {
            sample.horizonZone = -1; // Negative Time Event Horizon (backward CTC core)
        } else if (g00 >= -0.10) {
            sample.horizonZone = 0;  // Null Time Event Horizon / Cancellation Shell (g_00 ~ 0)
        } else {
            sample.horizonZone = 1;  // Positive Time Field (forward gravitational dilation)
        }

        // 6. Calibrated local proper gravitational acceleration a(x,y,z)/g_0 (in G's).
        //    Uses a softened Van Den Broeck screened potential calibrated so that at
        //    r = cabinProbeDistM (1.8 m) at 100% power, a = 1.85 G backward / ~1.61 G forward
        //    (both strictly within Titor's 1.5-2.0 G specification), dropping to 0 outside 22m x 6m.
        const double rSoft = 1.15;
        const double rRef = (cabinProbeDistM > 0.25) ? cabinProbeDistM : 1.8;
        const double refDenom = pow((rRef * rRef) + (rSoft * rSoft), 1.5);
        const double fAtRef = ComputeVanDenBroeckShapingFactor(
            ComputeC204EllipsoidalRadius(rRef, 0.0, 0.0, offsetXM, semiAxisA, semiAxisB, semiAxisC),
            wallSteepness
        );
        const double kRefScale = (fAtRef > 1.0e-6) ? (1.85 * refDenom / (rRef * fAtRef)) : 0.0;
        const double localDenom = pow((rCenter * rCenter) + (rSoft * rSoft), 1.5);
        const double accelG = cabinDriveScale * fEnv * kRefScale * (rCenter / localDenom);

        sample.properAccelG = accelG;
        // Exact GR static-observer gradient |grad g_00| = 2 a / c^2 (Turn 9 verbatim).
        const double accelSI = accelG * c204::kStandardGravitySI;
        sample.metricGradientInvM =
            (2.0 * accelSI) / (constants::kSpeedOfLightSI * constants::kSpeedOfLightSI);

        // 7. Figure 2 Item 5: 170-degree rearward X-ray radiation venting cone.
        sample.inXRayVentCone = IsPointInC204XRayVentCone(x, y, 0.0, xrayVentConeDeg);
        return sample;
    }

} // namespace umgebung::models::thinkers
