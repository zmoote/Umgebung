/**
 * @file ThinkerConstants.hpp
 * @brief Shared exact SI physical constants, geometric specifications, and
 *        CUDA `__host__ __device__` portability macro for all standalone
 *        Thinker modules (`models/thinkers/`) and `Unified::Synthesis`.
 *
 * @details
 * Centralizes constants cited verbatim across `submodules/research/Thinkers/`
 * so every standalone Thinker module and the combined `UnifiedSynthesisSolver`
 * evaluate identical IEEE-754 double-precision constants on both CPU and GPU.
 */

#pragma once

#if defined(__CUDACC__)
    #define UMGEBUNG_HD __host__ __device__
#else
    #define UMGEBUNG_HD
#endif

/**
 * @namespace umgebung::models::thinkers
 * @brief Standalone verbatim physics, device, and consciousness modules
 *        implemented directly from `submodules/research/Thinkers/`.
 */
namespace umgebung::models::thinkers {

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

} // namespace umgebung::models::thinkers
