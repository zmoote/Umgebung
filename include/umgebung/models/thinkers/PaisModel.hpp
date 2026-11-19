/**
 * @file PaisModel.hpp
 * @brief Verbatim quantitative equations for **Salvatore Pais**'s High-Energy
 *        Electromagnetic Field Generator (HEEMFG) & Inertial Mass Reduction
 *        Device (US Patent 10,144,532 B2).
 *
 * @details
 * Primary source: `submodules/research/Thinkers/Salvatore_Pais/US10144532.pdf`:
 *   - Eq. (1): Maximum HEEMFG Poynting energy flux:
 *     $$S_{\max} = f_G \frac{\sigma^2}{\varepsilon_0} (R_r \omega + R_v \nu + v_R)$$
 *   - Eq. (2): Accelerated harmonic vibration Poynting flux:
 *     $$S_{\max} = f_G \frac{\sigma^2}{\varepsilon_0} (R_v \nu^2 t_{\text{op}})$$
 *   - Vacuum-polarized effective inertial mass reduction factor:
 *     $$\frac{m_{\text{eff}}}{m_0} = \frac{1}{1 + \kappa_{\text{Pais}} S_{\max}} \in (0, 1]$$
 *     yielding gravitational/applied acceleration amplification
 *     $\mathbf{a}_{\text{HEEMFG}} = (1 + \kappa_{\text{Pais}} S_{\max})\,\mathbf{a}_0$.
 *     (Algebraic positivity and monotonicity bounds are formally verified in
 *     `formal/Umgebung/Unified/FlowerOfLife.lean`.)
 */

#pragma once

#include "umgebung/models/thinkers/ThinkerConstants.hpp"

namespace umgebung::models::thinkers {

    /**
     * @brief Computes Salvatore Pais's maximum HEEMFG Poynting energy flux
     *        $S_{\max} = f_G (\sigma^2 / \varepsilon_0) (R_r \omega + R_v \nu + v_R)$
     *        (`US10144532.pdf`, Equation 1).
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

    /**
     * @brief Computes the Pais effective inertial mass ratio
     *        $m_{\text{eff}} / m_0 = 1 / (1 + \kappa_{\text{Pais}} S_{\max}) \in (0, 1]$.
     *
     * @details Reduces monotonically toward `0` as vacuum-polarized electromagnetic
     *          energy density $\kappa_{\text{Pais}} S_{\max}$ increases, and equals
     *          exactly `1.0` when the HEEMFG cavity is inactive ($\kappa = 0$ or $S_{\max} = 0$).
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputePaisInertialMassFactor(
        double maxPoyntingFlux,
        double inertialReductionCoupling
    ) noexcept {
        const double product = maxPoyntingFlux * inertialReductionCoupling;
        if (product <= 0.0) {
            return 1.0;
        }
        return 1.0 / (1.0 + product);
    }

    /**
     * @brief Computes the resulting acceleration amplification factor
     *        $m_0 / m_{\text{eff}} = 1 + \kappa_{\text{Pais}} S_{\max} \ge 1.0$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputePaisAccelAmplification(
        double maxPoyntingFlux,
        double inertialReductionCoupling
    ) noexcept {
        const double product = maxPoyntingFlux * inertialReductionCoupling;
        return (product > 0.0) ? (1.0 + product) : 1.0;
    }

} // namespace umgebung::models::thinkers
