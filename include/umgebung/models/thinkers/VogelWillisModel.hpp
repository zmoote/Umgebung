/**
 * @file VogelWillisModel.hpp
 * @brief Verbatim quantitative specifications for **Marcel Vogel**, **Dan Willis**,
 *        and **Jen Han Eredyon**'s Phryll Crystal Generator (`PhryllGenInstructions.pdf`).
 *
 * @details
 * Primary source: `submodules/research/Thinkers/Dan_Willis/PhryllGenInstructions.pdf`:
 *   - **Crystal Cut Angles:** $51.84^\circ$ base termination and $60.0^\circ$ upper
 *     termination (exact pyramidal / Vogel-Eredyon geometry maximizing coherence).
 *   - **Orthogonal Copper/Silver Torsion Coils:** Driven with alternating $180^\circ$
 *     phase-shifted pulses at $f_0 = 4096\,\text{Hz}$ within the safe voltage
 *     window $V \in [20\,\text{V}, 60\,\text{V}]$.
 *   - **Crystal Saturation & "Singing" Threshold:** Cumulative driven energy
 *     $E_{\text{cum}} = P_{\text{drive}} \cdot t$ is monitored against
 *     `saturationThreshold`; when $E_{\text{cum}} / E_{\text{sat}} \ge 1.0$, the
 *     crystal reaches saturation/singing (risk of implosion per instructions).
 */

#pragma once

#include "umgebung/models/thinkers/ThinkerConstants.hpp"

#include <cmath>

namespace umgebung::models::thinkers {

    /**
     * @brief Returns `true` when the torsion-coil drive voltage lies within the
     *        documented safe operating window `[20.0 V, 60.0 V]` (`PhryllGenInstructions.pdf`).
     */
    UMGEBUNG_HD [[nodiscard]] inline bool IsVogelVoltageWithinSafeWindow(double driveVoltageV) noexcept {
        return (driveVoltageV >= constants::kPhryllSafeVoltageMinV) &&
               (driveVoltageV <= constants::kPhryllSafeVoltageMaxV);
    }

    /**
     * @brief Computes the dimensionless geometric resonance efficiency
     *        $\eta_{\text{geom}} \in (0, 1]$ measuring alignment to Vogel/Eredyon's
     *        $51.84^\circ$ base and $60.0^\circ$ top crystal termination angles.
     *
     * @details Evaluates to exactly `1.0` at $(51.84^\circ, 60.0^\circ)$ and decays
     *          smoothly as a Gaussian with angular bandwidth $\sigma_\theta = 10^\circ$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVogelGeometricResonanceFactor(
        double baseAngleDeg,
        double topAngleDeg
    ) noexcept {
        const double dBase = (baseAngleDeg - constants::kVogelBaseAngleDeg) / 10.0;
        const double dTop = (topAngleDeg - constants::kVogelTopAngleDeg) / 10.0;
        return exp(-0.5 * ((dBase * dBase) + (dTop * dTop)));
    }

    /**
     * @brief Computes normalized dimensionless torsion-coil excitation power
     *        $P_{\text{drive}} = (f / 4096\,\text{Hz}) \cdot (V / 20\,\text{V})^2 \cdot \eta_{\text{geom}}$.
     *
     * @details Evaluates to `1.0` at nominal baseline ($4096\,\text{Hz}$, $20\,\text{V}$,
     *          $51.84^\circ / 60.0^\circ$) and scales quadratically with coil voltage $V^2$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVogelTorsionDrivePower(
        double torsionFreqHz,
        double driveVoltageV,
        double geometricResonance
    ) noexcept {
        if (torsionFreqHz <= 0.0 || driveVoltageV <= 0.0 || geometricResonance <= 0.0) {
            return 0.0;
        }
        const double freqNorm = torsionFreqHz / constants::kPhryllTorsionFrequencyHz;
        const double voltNorm = driveVoltageV / constants::kPhryllSafeVoltageMinV;
        return freqNorm * (voltNorm * voltNorm) * geometricResonance;
    }

    /**
     * @brief Computes the symmetric pairwise Phryll resonant coupling scale between
     *        two entities carrying Phryll charges $Q_i, Q_j$:
     *        $F_{ij}^{\text{Phryll}} / r_{ij} = \kappa_{\text{Phryll}} P_{\text{drive}} (Q_i Q_j) / r_{ij}^3$.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVogelPhryllForceScale(
        double phryllCoupling,
        double drivePower,
        double phryllI,
        double phryllJ,
        double invDist3
    ) noexcept {
        return phryllCoupling * drivePower * (phryllI * phryllJ) * invDist3;
    }

    /**
     * @brief Computes the dimensionless crystal saturation ratio $E_{\text{cum}} / E_{\text{sat}}$.
     *        Values $\ge 1.0$ indicate crystal saturation / audible "singing" danger.
     */
    UMGEBUNG_HD [[nodiscard]] inline double ComputeVogelCrystalSaturationRatio(
        double cumulativeEnergy,
        double saturationThreshold
    ) noexcept {
        if (saturationThreshold <= 0.0 || cumulativeEnergy <= 0.0) {
            return 0.0;
        }
        return cumulativeEnergy / saturationThreshold;
    }

} // namespace umgebung::models::thinkers
