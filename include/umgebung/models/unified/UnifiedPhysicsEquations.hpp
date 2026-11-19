/**
 * @file UnifiedPhysicsEquations.hpp
 * @brief Verbatim quantitative equations and physical constants extracted
 *        directly from primary documents in `submodules/research/Thinkers/`.
 *
 * @details
 * Every formula in this header is annotated with its exact source paper,
 * equation number, physical units, and corresponding formal definition/theorem
 * in `formal/Umgebung/Unified/ThinkersModels.lean`:
 *
 *   1. **Nassim Haramein (2013)** — *Quantum Gravity and the Holographic Mass*,
 *      Physical Review & Research International 3(4): 270–292
 *      (`submodules/research/Thinkers/Nassim_Haramein/Haramein342013PRRI3363.pdf`):
 *        - Eq. (5):  Planck Spherical Unit (PSU) volume $V_{\ell s} = \frac{4}{3}\pi r_\ell^3$
 *        - Eq. (6):  PSU equatorial circular area $A_{\ell c} = \pi r_\ell^2$
 *        - Eq. (7):  Horizon surface PSU tiling count $\eta = A / A_{\ell c} = 4 r^2 / r_\ell^2$
 *        - Eq. (8):  Interior volume PSU oscillator count $\mathcal{R} = V / V_{\ell s} = r^3 / r_\ell^3$
 *        - Eq. (9)–(13): Holographic gravitational (Schwarzschild) mass $m_h = (\mathcal{R}/\eta) m_\ell$
 *        - Eq. (24)–(27): Holographic proton rest mass $m_p' = 2 (\eta/\mathcal{R}) m_\ell = 2\phi m_\ell$
 *
 *   2. **Chris Essonne** — *Gravity as the gradient of energy density of the vacuum*
 *      (`submodules/research/Thinkers/Chris_Essonne/research_papers_gravity_science_journal_8052.pdf`)
 *      & *The speed anomaly of stars at the periphery of spiral galaxies explained* (Feb 2025,
 *      `submodules/research/Thinkers/Chris_Essonne/research_papers_gravity_science_journal_10100.pdf`):
 *        - `8052.pdf` Eq. (21): Exterior vacuum pressure ratio $P(r)/P_0 = 1 - R_s / r$
 *        - `8052.pdf` Eq. (23): Vacuum refractive index $n(r) = 1 / \sqrt{1 - R_s / r}$
 *        - `8052.pdf` Eq. (27): Local speed of light $c'(r) = c\sqrt{1 - R_s / r}$
 *        - `8052.pdf` Eq. (35): Homogeneous interior pressure $P(r)/P_0 = \frac{R_s}{2R^3}(r^2 - R^2) + 1 - \frac{R_s}{R}$
 *        - `8052.pdf` Eq. (39): Zero central pressure critical radius $R = \frac{3}{2} R_s$
 *        - `8052.pdf` Eq. (43): True light-escape horizon radius $r = 2 R_s$
 *        - `10100.pdf` Eq. (7):  Inner galactic disk radial field $g_r(r) = \left(\frac{2GM}{h R_c^2} - \frac{g_z}{h}\right) r$
 *        - `10100.pdf` Eq. (11): Outer periphery radial field $g_r(r) = \frac{2GM - R_c^2 g_z}{h r}$ and flat velocity $v_{\text{flat}} = \sqrt{\frac{2GM - R_c^2 g_z}{h}}$
 *
 *   3. **Salvatore Pais** — *Craft Using an Inertial Mass Reduction Device*,
 *      US Patent 10,144,532 B2 (`submodules/research/Thinkers/Salvatore_Pais/US10144532.pdf`):
 *        - Eq. (1): Maximum HEEMFG Poynting flux $S_{\max} = f_G (\sigma^2/\varepsilon_0) (R_r \omega + R_v \nu + v_R)$
 *        - Eq. (2): Accelerated harmonic vibration Poynting flux $S_{\max} = f_G (\sigma^2/\varepsilon_0) (R_v \nu^2 t_{\text{op}})$
 *
 *   4. **Elena Danaan, Dan Willis & Marcel Vogel** — *Phryll Generator Crystal Construction Instructions*
 *      (`submodules/research/Thinkers/Dan_Willis/PhryllGenInstructions.pdf`):
 *        - Vogel/Eredyon facet angles ($51.84^\circ$ & $60.0^\circ$), orthogonal copper/silver torsion coil
 *          excitation frequency ($4096\,\text{Hz}$, $180^\circ$ phase shift, $20$–$60\,\text{V}$), and
 *          gravity-bound / vibrational-density non-linear local time rate.
 *
 * @note
 * Decorated with `__host__ __device__` when compiled by `nvcc` so the exact
 * same equations execute identically in host C++20 verification tests and in
 * CUDA FP64 simulation kernels.
 */

#pragma once

#include <cmath>

#if defined(__CUDACC__)
    #define UMGEBUNG_HD __host__ __device__
#else
    #define UMGEBUNG_HD
#endif

/**
 * @namespace umgebung::models::unified
 * @brief Pillar 2 verbatim unified physics & consciousness research models
 *        synthesized from `submodules/research/Thinkers/`.
 */
namespace umgebung::models::unified {

    /**
     * @namespace constants
     * @brief Exact SI physical and geometric constants cited in the Thinkers corpus.
     */
    namespace constants {
        /** @brief Mathematical constant $\pi$ in full IEEE-754 double precision. */
        inline constexpr double kPi = 3.141592653589793238462643383279502884;

        /** @brief Speed of light in vacuum $c$ ($\text{m}/\text{s}$, exact SI). */
        inline constexpr double kSpeedOfLightSI = 299792458.0;

        /** @brief Newtonian gravitational constant $G$ ($\text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$, CODATA). */
        inline constexpr double kGravitationalConstantSI = 6.67430e-11;

        /** @brief Vacuum electrical permittivity $\varepsilon_0$ ($\text{F}/\text{m}$, CODATA). */
        inline constexpr double kVacuumPermittivitySI = 8.8541878128e-12;

        /**
         * @brief Planck length $\ell_P$ in meters (CODATA value cited in Haramein 2013, p. 274:
         *        $1.616199 \times 10^{-33}\,\text{cm} = 1.616199 \times 10^{-35}\,\text{m}$).
         */
        inline constexpr double kPlanckLengthHaramein2013SI = 1.616199e-35;

        /**
         * @brief Planck mass $m_\ell$ in kilograms (CODATA value cited in Haramein 2013:
         *        $2.17651 \times 10^{-5}\,\text{g} = 2.17651 \times 10^{-8}\,\text{kg}$).
         */
        inline constexpr double kPlanckMassHaramein2013SI = 2.17651e-8;

        /** @brief Muonic hydrogen proton charge radius $r_p$ cited in Haramein 2013 Eq. (25) ($0.84184 \times 10^{-15}\,\text{m}$). */
        inline constexpr double kMuonicProtonRadiusSI = 0.84184e-15;

        /** @brief CODATA proton rest mass $m_p$ ($1.672622 \times 10^{-27}\,\text{kg}$). */
        inline constexpr double kCodataProtonMassSI = 1.672622e-27;

        /** @brief Vogel/Eredyon Phryll crystal base termination angle in degrees (`PhryllGenInstructions.pdf`). */
        inline constexpr double kVogelBaseAngleDeg = 51.84;

        /** @brief Vogel/Eredyon Phryll crystal upper termination angle in degrees (`PhryllGenInstructions.pdf`). */
        inline constexpr double kVogelTopAngleDeg = 60.0;

        /** @brief Orthogonal copper/silver torsion coil pulse frequency in Hz (`PhryllGenInstructions.pdf`). */
        inline constexpr double kPhryllTorsionFrequencyHz = 4096.0;

        /** @brief Minimum recommended safe torsion-coil drive voltage in Volts (`PhryllGenInstructions.pdf`). */
        inline constexpr double kPhryllSafeVoltageMinV = 20.0;

        /** @brief Maximum safe torsion-coil drive voltage prior to crystal saturation (`PhryllGenInstructions.pdf`). */
        inline constexpr double kPhryllSafeVoltageMaxV = 60.0;
    } // namespace constants

    // =========================================================================
    // 1. Nassim Haramein (2013) — Quantum Gravity & the Holographic Mass
    //    Source: submodules/research/Thinkers/Nassim_Haramein/Haramein342013PRRI3363.pdf
    // =========================================================================

    /**
     * @brief Computes the spherical volume $V_{\ell s}$ of a single Planck
     *        Spherical Unit (PSU) of diameter $\ell_P$ (`Haramein342013PRRI3363.pdf`, Eq. 5).
     *
     * @details $V_{\ell s} = \frac{4}{3}\pi r_\ell^3$ where $r_\ell = \ell_P / 2$.
     *
     * @param planckLength Diameter $\ell_P$ (default SI `kPlanckLengthHaramein2013SI`).
     * @return PSU spherical volume $V_{\ell s}$ in length$^3$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputePlanckSphericalVolume(
        double planckLength = constants::kPlanckLengthHaramein2013SI
    ) noexcept {
        const double lr = 0.5 * planckLength;
        return (4.0 / 3.0) * constants::kPi * lr * lr * lr;
    }

    /**
     * @brief Computes the equatorial circular tiling area $A_{\ell c}$ of a
     *        single PSU (`Haramein342013PRRI3363.pdf`, Eq. 6).
     *
     * @details $A_{\ell c} = \pi r_\ell^2$ where $r_\ell = \ell_P / 2$.
     *
     * @param planckLength Diameter $\ell_P$.
     * @return PSU equatorial area $A_{\ell c}$ in length$^2$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputePlanckEquatorialArea(
        double planckLength = constants::kPlanckLengthHaramein2013SI
    ) noexcept {
        const double lr = 0.5 * planckLength;
        return constants::kPi * lr * lr;
    }

    /**
     * @brief Computes the number of equatorial PSU circular areas tiling a
     *        spherical horizon of radius $r$ (`Haramein342013PRRI3363.pdf`, Eq. 7).
     *
     * @details $\eta(r) = \frac{4\pi r^2}{\pi r_\ell^2} = \frac{4 r^2}{r_\ell^2} = \frac{16 r^2}{\ell_P^2}$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeSurfacePSUCount(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI
    ) noexcept {
        const double area = 4.0 * constants::kPi * radius * radius;
        return area / ComputePlanckEquatorialArea(planckLength);
    }

    /**
     * @brief Computes the number of interior spherical PSU oscillators within a
     *        sphere of radius $r$ (`Haramein342013PRRI3363.pdf`, Eq. 8).
     *
     * @details $\mathcal{R}(r) = \frac{\frac{4}{3}\pi r^3}{\frac{4}{3}\pi r_\ell^3} = \frac{r^3}{r_\ell^3} = \frac{8 r^3}{\ell_P^3}$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVolumePSUCount(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI
    ) noexcept {
        const double vol = (4.0 / 3.0) * constants::kPi * radius * radius * radius;
        return vol / ComputePlanckSphericalVolume(planckLength);
    }

    /**
     * @brief Computes Haramein's fundamental holographic surface-to-volume ratio
     *        $\phi(r) = \eta(r) / \mathcal{R}(r) = 2 \ell_P / r$
     *        (`Haramein342013PRRI3363.pdf`, Section 2–3).
     *
     * @details
     * Computed via the algebraically reduced closed form $2\ell_P / r$ to avoid
     * IEEE-754 floating-point overflow when $r$ is at astrophysical scales
     * (where $\mathcal{R} \sim 10^{118} - 10^{180}$). Formally proven identical
     * to $\eta(r)/\mathcal{R}(r)$ in `formal/Umgebung/Unified/ThinkersModels.lean`.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHolographicRatioPhi(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI
    ) noexcept {
        return (radius > 0.0) ? ((2.0 * planckLength) / radius) : 0.0;
    }

    /**
     * @brief Computes Haramein's holographic gravitational (Schwarzschild) mass
     *        $m_h(r) = (\mathcal{R}/\eta) m_\ell = m_\ell / \phi(r) = \frac{r m_\ell}{2 \ell_P}$
     *        (`Haramein342013PRRI3363.pdf`, Eq. 9–13).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHolographicSchwarzschildMass(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI,
        double planckMass = constants::kPlanckMassHaramein2013SI
    ) noexcept {
        return (planckLength > 0.0) ? ((radius * planckMass) / (2.0 * planckLength)) : 0.0;
    }

    /**
     * @brief Computes Haramein's holographic rest mass $m_p'(r) = 2 \phi(r) m_\ell = 2 (\eta/\mathcal{R}) m_\ell$
     *        (`Haramein342013PRRI3363.pdf`, Eq. 24–27).
     *
     * @details
     * Evaluated at the 2010 muonic proton charge radius $r_p = 0.84184 \times 10^{-15}\,\text{m}$,
     * yields $1.67142 \times 10^{-27}\,\text{kg}$ (~0.07% from CODATA $1.67262 \times 10^{-27}\,\text{kg}$).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHolographicProtonMass(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI,
        double planckMass = constants::kPlanckMassHaramein2013SI
    ) noexcept {
        return 2.0 * ComputeHolographicRatioPhi(radius, planckLength) * planckMass;
    }

    // =========================================================================
    // 2. Chris Essonne — Vacuum Energy Density Gradient & Spiral Galaxy Vortex
    //    Sources: research_papers_gravity_science_journal_8052.pdf
    //             research_papers_gravity_science_journal_10100.pdf
    // =========================================================================

    /**
     * @brief Computes the classical Schwarzschild radius $R_s = 2GM/c^2$ (`8052.pdf`, Eq. 20).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeSchwarzschildRadius(
        double mass,
        double G = constants::kGravitationalConstantSI,
        double c = constants::kSpeedOfLightSI
    ) noexcept {
        return (c > 0.0) ? ((2.0 * G * mass) / (c * c)) : 0.0;
    }

    /**
     * @brief Computes Chris Essonne's exterior vacuum pressure ratio
     *        $P(r)/P_0 = 1 - R_s / r$ (`8052.pdf`, Eq. 21).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneVacuumPressureRatio(
        double radius,
        double schwarzschildRadius
    ) noexcept {
        return (radius > 0.0) ? (1.0 - (schwarzschildRadius / radius)) : 0.0;
    }

    /**
     * @brief Computes Chris Essonne's local speed of light $c'(r) = c\sqrt{1 - R_s / r}$
     *        (`8052.pdf`, Eq. 27). Clamps negative radicands inside $r < R_s$ to `0.0`.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneLocalLightSpeed(
        double radius,
        double schwarzschildRadius,
        double c = constants::kSpeedOfLightSI
    ) noexcept {
        const double ratio = ComputeEssonneVacuumPressureRatio(radius, schwarzschildRadius);
        return (ratio > 0.0) ? (c * sqrt(ratio)) : 0.0;
    }

    /**
     * @brief Computes Chris Essonne's vacuum optical refractive index
     *        $n(r) = 1 / \sqrt{1 - R_s / r}$ (`8052.pdf`, Eq. 23).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneRefractiveIndex(
        double radius,
        double schwarzschildRadius
    ) noexcept {
        const double ratio = ComputeEssonneVacuumPressureRatio(radius, schwarzschildRadius);
        return (ratio > 0.0) ? (1.0 / sqrt(ratio)) : 0.0;
    }

    /**
     * @brief Computes Chris Essonne's interior vacuum pressure ratio inside a
     *        homogeneous sphere of radius $R$ (`8052.pdf`, Eq. 35).
     *
     * @details
     * $$P(r)/P_0 = \frac{R_s}{2 R^3}(r^2 - R^2) + 1 - \frac{R_s}{R}$$
     * Evaluates to `0.0` at the center ($r = 0$) precisely when $R = \frac{3}{2} R_s$ (`Eq. 39`).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneInteriorPressureRatio(
        double r,
        double bodyRadius,
        double schwarzschildRadius
    ) noexcept {
        if (bodyRadius <= 0.0) {
            return 0.0;
        }
        const double r3 = bodyRadius * bodyRadius * bodyRadius;
        return ((schwarzschildRadius / (2.0 * r3)) * ((r * r) - (bodyRadius * bodyRadius))) +
               1.0 - (schwarzschildRadius / bodyRadius);
    }

    /**
     * @brief Computes Chris Essonne's 2D spiral-galaxy aether-vortex inward radial
     *        gravitational acceleration magnitude $g_r(r)$ (`10100.pdf`, Eq. 7 & 11).
     *
     * @details
     * For a flat cylindrical galactic core of mass $M$, critical radius $R_c$,
     * scale height $h$, and inner vertical field $g_z$ (where $\rho_m = M / (\pi R_c^2 h)$):
     *   - Inner region ($r \le R_c$, Eq. 7):
     *     $$g_r(r) = \left(2\pi \rho_m G - \frac{g_z}{h}\right) r = \frac{2GM - R_c^2 g_z}{h R_c^2} r$$
     *   - Outer periphery ($r > R_c$, Eq. 11, where 2D aether vortex suppresses $g_z = 0$):
     *     $$g_r(r) = \frac{2GM - R_c^2 g_z}{h \cdot r}$$
     *
     * @param cylindricalRadius Cylindrical radial distance $r = \sqrt{x^2 + y^2}$ from core.
     * @param coreMass          Galactic core mass $M$.
     * @param G                 Gravitational constant $G$.
     * @param criticalRadius    Critical galactic disk radius $R_c > 0$.
     * @param scaleHeight       Disk scale height $h > 0$.
     * @param verticalFieldGz   Inner vertical gravitational component $g_z$.
     * @return Inward radial acceleration magnitude $g_r(r) \ge 0$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneAetherVortexRadialAccel(
        double cylindricalRadius,
        double coreMass,
        double G,
        double criticalRadius,
        double scaleHeight,
        double verticalFieldGz
    ) noexcept {
        if (criticalRadius <= 0.0 || scaleHeight <= 0.0 || cylindricalRadius <= 0.0) {
            return 0.0;
        }

        const double numerator = (2.0 * G * coreMass) - (criticalRadius * criticalRadius * verticalFieldGz);
        if (numerator <= 0.0) {
            return 0.0;
        }

        if (cylindricalRadius <= criticalRadius) {
            return (numerator / (scaleHeight * criticalRadius * criticalRadius)) * cylindricalRadius;
        }
        return numerator / (scaleHeight * cylindricalRadius);
    }

    /**
     * @brief Computes Chris Essonne's predicted constant flat orbital speed
     *        $v_{\text{flat}} = \sqrt{(2GM - R_c^2 g_z) / h}$ for periphery stars
     *        ($r > R_c$) without dark matter (`10100.pdf`, Eq. 11).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneFlatRotationVelocity(
        double coreMass,
        double G,
        double criticalRadius,
        double scaleHeight,
        double verticalFieldGz
    ) noexcept {
        if (scaleHeight <= 0.0) {
            return 0.0;
        }
        const double vSq = ((2.0 * G * coreMass) - (criticalRadius * criticalRadius * verticalFieldGz)) / scaleHeight;
        return (vSq > 0.0) ? sqrt(vSq) : 0.0;
    }

    /**
     * @brief Computes the exact $C^1$-continuous radial potential energy per unit
     *        mass $\Phi_{\text{vortex}}(r)$ corresponding to
     *        `ComputeEssonneAetherVortexRadialAccel` ($\Phi(R_c) = 0$).
     *
     * @details
     *   - Inner ($r \le R_c$): $\Phi(r) = \frac{2GM - R_c^2 g_z}{2 h R_c^2} (r^2 - R_c^2)$
     *   - Outer ($r > R_c$):   $\Phi(r) = \frac{2GM - R_c^2 g_z}{h} \ln(r / R_c)$
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeEssonneAetherVortexPotential(
        double cylindricalRadius,
        double coreMass,
        double G,
        double criticalRadius,
        double scaleHeight,
        double verticalFieldGz
    ) noexcept {
        if (criticalRadius <= 0.0 || scaleHeight <= 0.0 || cylindricalRadius <= 0.0) {
            return 0.0;
        }

        const double numerator = (2.0 * G * coreMass) - (criticalRadius * criticalRadius * verticalFieldGz);
        if (numerator <= 0.0) {
            return 0.0;
        }

        if (cylindricalRadius <= criticalRadius) {
            const double rcSq = criticalRadius * criticalRadius;
            return (numerator / (2.0 * scaleHeight * rcSq)) * ((cylindricalRadius * cylindricalRadius) - rcSq);
        }
        return (numerator / scaleHeight) * log(cylindricalRadius / criticalRadius);
    }

    // =========================================================================
    // 3. Salvatore Pais — High-Energy Electromagnetic Field Generator (HEEMFG)
    //    Source: submodules/research/Thinkers/Salvatore_Pais/US10144532.pdf
    // =========================================================================

    /**
     * @brief Computes Salvatore Pais's maximum HEEMFG Poynting energy flux
     *        $S_{\max} = f_G (\sigma^2 / \varepsilon_0) (R_r \omega + R_v \nu + v_R)$
     *        (`US10144532.pdf`, Equation 1).
     *
     * @param geometricFactor   Shape factor $f_G$ (`1.0` for a disc configuration).
     * @param surfaceChargeDens Surface charge density $\sigma$ ($\text{C}/\text{m}^2$).
     * @param rotRadius         Disc rotation radius $R_r$ ($\text{m}$).
     * @param angularFreqRadS   Axial rotation angular frequency $\omega$ ($\text{rad}/\text{s}$).
     * @param vibAmplitude      Harmonic vibration amplitude $R_v$ ($\text{m}$).
     * @param vibFreqHz         Vibration frequency $\nu$ ($\text{Hz}$).
     * @param translationSpeed  Curvilinear translation speed $v_R$ ($\text{m}/\text{s}$).
     * @param permittivity      Vacuum permittivity $\varepsilon_0$ ($\text{F}/\text{m}$).
     * @return Maximum Poynting energy flux $S_{\max}$ ($\text{W}/\text{m}^2$).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputePaisMaxPoyntingFlux(
        double geometricFactor,
        double surfaceChargeDens,
        double rotRadius,
        double angularFreqRadS,
        double vibAmplitude,
        double vibFreqHz,
        double translationSpeed,
        double permittivity = constants::kVacuumPermittivitySI
    ) noexcept {
        if (permittivity <= 0.0) {
            return 0.0;
        }
        return geometricFactor *
               ((surfaceChargeDens * surfaceChargeDens) / permittivity) *
               ((rotRadius * angularFreqRadS) + (vibAmplitude * vibFreqHz) + translationSpeed);
    }

    /**
     * @brief Computes Salvatore Pais's accelerated harmonic vibration Poynting flux
     *        $S_{\max} = f_G (\sigma^2 / \varepsilon_0) (R_v \nu^2 t_{\text{op}})$
     *        (`US10144532.pdf`, Equation 2).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputePaisAcceleratedPoyntingFlux(
        double geometricFactor,
        double surfaceChargeDens,
        double vibAmplitude,
        double vibFreqHz,
        double operationalTimeSec,
        double permittivity = constants::kVacuumPermittivitySI
    ) noexcept {
        if (permittivity <= 0.0) {
            return 0.0;
        }
        return geometricFactor *
               ((surfaceChargeDens * surfaceChargeDens) / permittivity) *
               (vibAmplitude * vibFreqHz * vibFreqHz * operationalTimeSec);
    }

    // =========================================================================
    // 4. Elena Danaan / Dan Willis / Marcel Vogel — Non-Linear Time & Phryll
    //    Sources: PhryllGenInstructions.pdf, Personal/Audio Script Project Umgebun.txt
    // =========================================================================

    /**
     * @brief Computes an entity's local proper-time rate $d\tau / dt$ coupled to
     *        Essonne's local vacuum pressure ratio $P/P_0$ and Elena Danaan's
     *        vibrational density scale $D / D_0$.
     *
     * @details
     * Models the synthesis that linear time slows under deep gravitational
     * vacuum pressure depletion ($\sqrt{P/P_0} = c'/c$, matching Schwarzschild
     * gravitational time dilation) and shifts across vibrational density octaves
     * ($D / D_0$). When `nonLinearTimeEnabled == false`, callers use `1.0`.
     *
     * @param vacuumPressureRatio Local ratio $P_i / P_0 \in [0, 1]$.
     * @param vibrationalDensity  Entity's vibrational density $D_i > 0$.
     * @param baseDensity         Universe baseline vibrational density $D_0 > 0$.
     * @return Dimensionless proper-time rate $d\tau_i / dt \ge 0$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeLocalTimeRate(
        double vacuumPressureRatio,
        double vibrationalDensity,
        double baseDensity
    ) noexcept {
        const double gravTimeFactor = (vacuumPressureRatio > 0.0) ? sqrt(vacuumPressureRatio) : 0.0;
        const double densityFactor =
            (baseDensity > 0.0 && vibrationalDensity > 0.0)
                ? (vibrationalDensity / baseDensity)
                : 1.0;
        return gravTimeFactor * densityFactor;
    }

} // namespace umgebung::models::unified
