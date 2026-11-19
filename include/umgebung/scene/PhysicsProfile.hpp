/**
 * @file PhysicsProfile.hpp
 * @brief Declarative per-universe physics profile configuration supporting
 *        Mainstream Physics baselines, Unified Research Models, and custom
 *        A/B compositions inside `.umgebung` scene files.
 *
 * @details
 * Every universe in Umgebung maps to its own `.umgebung` JSON scene file
 * declaring both its initial state and its governing `PhysicsProfileConfig`.
 * Keeping Pillar 1 (`MainstreamGravityParams`) and Pillar 2
 * (`UnifiedFieldParams`) parameters cleanly structured inside one schema allows
 * controlled side-by-side A/B testing on identical initial conditions.
 *
 * Source provenance for Pillar 2 (`UnifiedFieldParams`):
 *   - **Nassim Haramein** (`Haramein342013PRRI3363.pdf`): Planck Spherical Unit
 *     (PSU) holographic mass ratios $\phi = \eta / \mathcal{R}$.
 *   - **Chris Essonne** (`research_papers_gravity_science_journal_8052.pdf` &
 *     `10100.pdf`): Vacuum energy density / aether pressure gradient gravity and
 *     2D galactic aether-vortex flat rotation curve parameters ($R_c, h, g_z$).
 *   - **Salvatore Pais** (`US10144532.pdf`): High-Energy Electromagnetic Field
 *     Generator (HEEMFG) Poynting flux / vacuum polarization coupling.
 *   - **Elena Danaan, Dan Willis & Marcel Vogel** (`PhryllGenInstructions.pdf`):
 *     Love binding medium coupling, Phryll resonant field, vibrational density,
 *     and non-linear gravity-coupled local time fields.
 */

#pragma once

#include <cstdint>
#include <string>

/**
 * @namespace umgebung::scene
 * @brief Multiverse/Universe domain hierarchy, `.umgebung` JSON serialization,
 *        and per-universe physics profile descriptors (zero GPU or GUI deps).
 */
namespace umgebung::scene {

    /**
     * @enum PhysicsProfileType
     * @brief Identifies which governing theoretical framework animates a universe.
     */
    enum class PhysicsProfileType : std::uint8_t {
        /** @brief Pillar 1: Verified textbook physics solvers (Newtonian/PN gravity, Maxwell EM, etc.). */
        Mainstream = 0,
        /** @brief Pillar 2: Verbatim unified reality model synthesized from `submodules/research/Thinkers/`. */
        Unified = 1,
        /** @brief Custom A/B composition mixing baseline mainstream forces with selected unified terms. */
        Custom = 2
    };

    /**
     * @enum IntegratorType
     * @brief Numerical time-integration scheme selected for the universe's dynamical evolution.
     */
    enum class IntegratorType : std::uint8_t {
        /** @brief 2nd-order symplectic, time-reversible Kick-Drift-Kick Velocity Verlet. */
        VelocityVerlet = 0,
        /** @brief 1st-order explicit Euler integrator (non-symplectic pedagogical baseline). */
        EulerExplicit = 1
    };

    /**
     * @struct MainstreamGravityParams
     * @brief Governing parameters for textbook Newtonian pairwise N-body gravity.
     *
     * @details
     * Pairwise acceleration on body $i$:
     * $$\mathbf{a}_i = G \sum_{j \neq i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{\left(\|\mathbf{r}_j - \mathbf{r}_i\|^2 + \varepsilon^2\right)^{3/2}}$$
     */
    struct MainstreamGravityParams {
        /**
         * @brief Gravitational constant $G$.
         *        Defaults to CODATA SI $6.67430 \times 10^{-11}\ \text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$;
         *        set to `1.0` for dimensionless normalized N-body benchmarks.
         */
        double gravitationalConstant{6.67430e-11};

        /**
         * @brief Plummer gravitational softening length $\varepsilon \ge 0$ (in scene length units).
         *        Set to `0.0` for exact unsoftened analytical benchmarks.
         */
        double softeningLength{0.0};
    };

    /**
     * @struct UnifiedFieldParams
     * @brief Verbatim quantitative parameters for Pillar 2 (`models/unified/`)
     *        synthesized from `submodules/research/Thinkers/`.
     */
    struct UnifiedFieldParams {
        // ---------------------------------------------------------------------
        // Haramein (2013) Holographic Vacuum & Scaling Parameters
        // ---------------------------------------------------------------------

        /** @brief Ambient vacuum Planck energy density $\rho_{\text{vac}}$ ($\text{kg}/\text{m}^3$ or normalized). */
        double harmonicVacuumDensity{5.155e96};

        // ---------------------------------------------------------------------
        // Essonne (2025) Vacuum Pressure Gradient & Galactic Aether Vortex
        // ---------------------------------------------------------------------

        /**
         * @brief Reference speed of light $c$ in scene length/time units for
         *        Essonne's vacuum pressure ratio $P(r)/P_0 = 1 - 2GM/(c^2 r)$
         *        and local light speed $c'(r) = c\sqrt{1 - R_s / r}$ (`8052.pdf` Eq. 21, 27).
         */
        double referenceLightSpeed{299792458.0};

        /**
         * @brief Enables Chris Essonne's 2D galactic aether-vortex radial field
         *        (`research_papers_gravity_science_journal_10100.pdf` Eq. 7 & 11)
         *        replacing isotropic $1/r^2$ central-core falloff at galactic scales.
         */
        bool aetherVortexEnabled{false};

        /** @brief Critical galactic disk radius $R_c$ (`10100.pdf` Eq. 7, 11). */
        double aetherVortexCriticalRadius{1.0};

        /** @brief Galactic disk scale height $h$ (`10100.pdf` Eq. 7, 11; must be $> 0$ when enabled). */
        double aetherVortexScaleHeight{1.0};

        /** @brief Uniform vertical gravitational field $g_z$ over the inner disk $r \le R_c$ (`10100.pdf` Eq. 6–11). */
        double aetherVortexVerticalFieldGz{0.0};

        // ---------------------------------------------------------------------
        // Danaan / Willis / Vogel — Love, Phryll & Non-Linear Time Fields
        // ---------------------------------------------------------------------

        /** @brief Pairwise Love binding-medium coherence coupling constant (antisymmetric $\mathbf{F}_{ij} = -\mathbf{F}_{ji}$). */
        double loveBindingCoupling{0.0};

        /** @brief Ambient Phryll life-force / crystal vortex field amplitude (`PhryllGenInstructions.pdf`). */
        double phryllFieldAmplitude{0.0};

        /** @brief Baseline vibrational density octave for the universe (default `3.0` = 3rd density). */
        double baseVibrationalDensity{3.0};

        /** @brief Enables gravity/vacuum-pressure & vibrational-density coupled local time rate $\Delta\tau_i$. */
        bool nonLinearTimeEnabled{false};
    };

    /**
     * @struct PhysicsProfileConfig
     * @brief Complete declarative physics configuration embedded in a `UniverseScene`.
     */
    struct PhysicsProfileConfig {
        /** @brief Active theoretical profile (`Mainstream`, `Unified`, or `Custom`). */
        PhysicsProfileType type{PhysicsProfileType::Mainstream};

        /** @brief Selected numerical time integrator (`VelocityVerlet` default). */
        IntegratorType integrator{IntegratorType::VelocityVerlet};

        /** @brief Unit convention label (`"SI"` or `"Normalized"`). */
        std::string unitSystem{"Normalized"};

        /** @brief Integration timestep $\Delta t$ in scene time units (must be `> 0`). */
        double timestep{1.0e-3};

        /** @brief Default step count executed in headless batch mode (`--steps` overrides). */
        std::uint64_t defaultSteps{1000ULL};

        /** @brief Mainstream Newtonian gravity parameters. */
        MainstreamGravityParams mainstreamGravity{};

        /** @brief Unified Research Model parameters (used by `Unified` / `Custom` profiles). */
        UnifiedFieldParams unifiedField{};
    };

    /**
     * @brief Converts a `PhysicsProfileType` enum to its canonical `.umgebung` JSON string.
     */
    [[nodiscard]] inline const char* PhysicsProfileTypeToString(PhysicsProfileType type) noexcept {
        switch (type) {
            case PhysicsProfileType::Mainstream:
                return "Mainstream";
            case PhysicsProfileType::Unified:
                return "Unified";
            case PhysicsProfileType::Custom:
                return "Custom";
        }
        return "Mainstream";
    }

    /**
     * @brief Parses a `.umgebung` JSON profile string into `PhysicsProfileType`.
     */
    [[nodiscard]] inline bool PhysicsProfileTypeFromString(
        const std::string& text,
        PhysicsProfileType& outVal
    ) noexcept {
        if (text == "Mainstream" || text == "mainstream") {
            outVal = PhysicsProfileType::Mainstream;
            return true;
        }
        if (text == "Unified" || text == "unified") {
            outVal = PhysicsProfileType::Unified;
            return true;
        }
        if (text == "Custom" || text == "custom") {
            outVal = PhysicsProfileType::Custom;
            return true;
        }
        return false;
    }

    /**
     * @brief Converts an `IntegratorType` enum to its canonical `.umgebung` JSON string.
     */
    [[nodiscard]] inline const char* IntegratorTypeToString(IntegratorType integrator) noexcept {
        switch (integrator) {
            case IntegratorType::VelocityVerlet:
                return "VelocityVerlet";
            case IntegratorType::EulerExplicit:
                return "EulerExplicit";
        }
        return "VelocityVerlet";
    }

    /**
     * @brief Parses a `.umgebung` JSON integrator string into `IntegratorType`.
     */
    [[nodiscard]] inline bool IntegratorTypeFromString(
        const std::string& text,
        IntegratorType& outVal
    ) noexcept {
        if (text == "VelocityVerlet" || text == "velocity_verlet") {
            outVal = IntegratorType::VelocityVerlet;
            return true;
        }
        if (text == "EulerExplicit" || text == "euler_explicit") {
            outVal = IntegratorType::EulerExplicit;
            return true;
        }
        return false;
    }

} // namespace umgebung::scene
