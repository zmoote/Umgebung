/**
 * @file HarameinModel.hpp
 * @brief Verbatim quantitative equations for **Nassim Haramein**'s holographic
 *        mass (Planck Spherical Unit tiling) and Haramein-Rauscher spacetime
 *        torque / Coriolis dynamics.
 *
 * @details
 * Primary sources in `submodules/research/Thinkers/Nassim_Haramein/`:
 *   1. `Haramein342013PRRI3363.pdf` (*Quantum Gravity and the Holographic Mass*,
 *      PRRI 3(4): 270–292, 2013):
 *        - Eq. (5):  PSU spherical volume $V_{\ell s} = \frac{4}{3}\pi r_\ell^3$
 *        - Eq. (6):  PSU equatorial tiling area $A_{\ell c} = \pi r_\ell^2$
 *        - Eq. (7):  Horizon surface PSU count $\eta = A / A_{\ell c} = 4 r^2 / r_\ell^2$
 *        - Eq. (8):  Interior volume PSU count $\mathcal{R} = V / V_{\ell s} = r^3 / r_\ell^3$
 *        - Eq. (9)–(13): Holographic Schwarzschild mass $m_h = (\mathcal{R}/\eta) m_\ell$
 *        - Eq. (24)–(27): Holographic proton rest mass $m_p' = 2 (\eta/\mathcal{R}) m_\ell = 2 \phi m_\ell$
 *   2. `Scale-Unification` / `Collective-Coherent-Oscillation`:
 *        - Haramein-Rauscher spacetime torque Coriolis acceleration
 *          $\mathbf{a}_{\text{Cor}} = -2 \boldsymbol{\Omega} \times \mathbf{v}$
 *          which is strictly perpendicular to $\mathbf{v}$ ($\mathbf{a}_{\text{Cor}} \cdot \mathbf{v} = 0$)
 *          and therefore performs zero mechanical work while imparting frame-dragging spin.
 */

#pragma once

#include "umgebung/models/thinkers/ThinkerConstants.hpp"

namespace umgebung::models::thinkers {

    /**
     * @brief Computes the spherical volume $V_{\ell s}$ of a single Planck
     *        Spherical Unit (PSU) of diameter $\ell_P$ (`Haramein342013PRRI3363.pdf`, Eq. 5).
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
     * @details Closed-form reduction avoids IEEE-754 overflow at astrophysical radii;
     *          formally proven identical to $\eta/\mathcal{R}$ in `ThinkersModels.lean`.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHolographicRatioPhi(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI
    ) noexcept {
        return (radius > 0.0) ? ((2.0 * planckLength) / radius) : 0.0;
    }

    /**
     * @brief Computes Haramein's holographic gravitational (Schwarzschild) mass
     *        $m_h(r) = (\mathcal{R}/\eta) m_\ell = r m_\ell / (2 \ell_P)$
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
     * @brief Computes Haramein's holographic rest mass $m_p'(r) = 2 \phi(r) m_\ell$
     *        (`Haramein342013PRRI3363.pdf`, Eq. 24–27).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHolographicProtonMass(
        double radius,
        double planckLength = constants::kPlanckLengthHaramein2013SI,
        double planckMass = constants::kPlanckMassHaramein2013SI
    ) noexcept {
        return 2.0 * ComputeHolographicRatioPhi(radius, planckLength) * planckMass;
    }

    /**
     * @brief X-component of Haramein-Rauscher spacetime-torque Coriolis acceleration
     *        $\mathbf{a}_{\text{Cor}} = -2 \boldsymbol{\Omega} \times \mathbf{v}$
     *        for $\boldsymbol{\Omega} = (0, 0, \Omega_z)$: $a_{\text{Cor}, x} = +2 \Omega_z v_y$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHarameinCoriolisAccelX(
        double vy,
        double omegaZ
    ) noexcept {
        return 2.0 * omegaZ * vy;
    }

    /**
     * @brief Y-component of Haramein-Rauscher spacetime-torque Coriolis acceleration
     *        $\mathbf{a}_{\text{Cor}} = -2 \boldsymbol{\Omega} \times \mathbf{v}$
     *        for $\boldsymbol{\Omega} = (0, 0, \Omega_z)$: $a_{\text{Cor}, y} = -2 \Omega_z v_x$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeHarameinCoriolisAccelY(
        double vx,
        double omegaZ
    ) noexcept {
        return -2.0 * omegaZ * vx;
    }

} // namespace umgebung::models::thinkers
