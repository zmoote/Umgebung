/**
 * @file EssonneModel.hpp
 * @brief Verbatim quantitative equations for **Chris Essonne**'s vacuum energy
 *        density gradient gravity and 2D spiral-galaxy aether-vortex model.
 *
 * @details
 * Primary sources in `submodules/research/Thinkers/Chris_Essonne/`:
 *   1. `research_papers_gravity_science_journal_8052.pdf` (*Gravity as the
 *      gradient of energy density of the vacuum*):
 *        - Eq. (20): Schwarzschild radius $R_s = 2GM/c^2$
 *        - Eq. (21): Exterior vacuum pressure ratio $P(r)/P_0 = 1 - R_s / r$
 *        - Eq. (23): Vacuum optical refractive index $n(r) = 1 / \sqrt{1 - R_s / r}$
 *        - Eq. (27): Local speed of light $c'(r) = c\sqrt{1 - R_s / r}$
 *        - Eq. (35): Homogeneous interior pressure $P(r)/P_0 = \frac{R_s}{2R^3}(r^2 - R^2) + 1 - \frac{R_s}{R}$
 *        - Eq. (39): Zero central pressure critical radius $R = \frac{3}{2} R_s$
 *        - Eq. (43): True light-escape horizon radius $r = 2 R_s$
 *   2. `research_papers_gravity_science_journal_10100.pdf` (*The speed anomaly
 *      of stars at the periphery of spiral galaxies explained*, Feb 2025):
 *        - Eq. (7):  Inner galactic disk radial field $g_r(r) = \frac{2GM - R_c^2 g_z}{h R_c^2} r$
 *        - Eq. (11): Outer periphery radial field $g_r(r) = \frac{2GM - R_c^2 g_z}{h r}$
 *                    and constant flat speed $v_{\text{flat}} = \sqrt{(2GM - R_c^2 g_z)/h}$
 *
 * Formal Lean 4 counterparts live in `formal/Umgebung/Unified/ThinkersModels.lean`.
 */

#pragma once

#include "umgebung/models/thinkers/ThinkerConstants.hpp"

#include <cmath>

namespace umgebung::models::thinkers {

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
     * @details Evaluates to `0.0` at $r = 0$ precisely when $R = \frac{3}{2} R_s$ (`Eq. 39`).
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

} // namespace umgebung::models::thinkers
