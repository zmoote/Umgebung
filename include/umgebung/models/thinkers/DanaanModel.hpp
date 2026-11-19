/**
 * @file DanaanModel.hpp
 * @brief Verbatim quantitative equations for **Elena Danaan**'s cosmological
 *        framework: Love binding coherence force, vibrational density octaves,
 *        and gravity-bound non-linear local proper time.
 *
 * @details
 * Primary sources:
 *   - `submodules/research/Personal/### ## Audio Script Project Umgebun.txt`
 *   - `submodules/research/Thinkers/Elena_Danaan/`
 *
 * Key properties:
 *   - **Love Binding Coherence Force:** Strictly symmetric coupling scalar
 *     $\kappa_{\text{love}} C_i C_j (Q_i + Q_j) / r_{ij}^3$, guaranteeing
 *     $\mathbf{F}_{ij}^{\text{love}} = -\mathbf{F}_{ji}^{\text{love}}$ and exact
 *     total linear momentum conservation.
 *   - **Non-Linear Local Proper Time:** Linear time slows inside gravitational
 *     vacuum depletion ($\sqrt{P_i/P_0}$) and scales across vibrational density
 *     octaves ($D_i / D_0$).
 */

#pragma once

#include "umgebung/models/thinkers/ThinkerConstants.hpp"

#include <cmath>

namespace umgebung::models::thinkers {

    /**
     * @brief Computes an entity's local proper-time rate $d\tau / dt$ coupled to
     *        gravitational vacuum depletion $P/P_0$ and Elena Danaan's vibrational
     *        density scale $D / D_0$.
     *
     * @param vacuumPressureRatio Local ratio $P_i / P_0 \in [0, 1]$.
     * @param vibrationalDensity  Entity vibrational density $D_i > 0$.
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

    /**
     * @brief Computes the symmetric scalar factor of Elena Danaan's Love binding
     *        coherence force between entities $i$ and $j$:
     *        $F_{ij}^{\text{love}} / \|\mathbf{r}_j - \mathbf{r}_i\| =
     *        \kappa_{\text{love}} \, C_i C_j (Q_i + Q_j) / r_{ij}^3$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeDanaanLoveForceScale(
        double loveCoupling,
        double coherenceI,
        double coherenceJ,
        double phryllI,
        double phryllJ,
        double invDist3
    ) noexcept {
        return loveCoupling * (coherenceI * coherenceJ) * (phryllI + phryllJ) * invDist3;
    }

} // namespace umgebung::models::thinkers
