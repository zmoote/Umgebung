/**
 * @file PhysicsProfile.hpp
 * @brief Declarative per-universe physics profile configuration supporting
 *        Mainstream Physics baselines, Unified Research Models, and custom
 *        A/B compositions inside `.umgebung` scene files.
 *
 * @details
 * Every universe in Umgebung maps to its own `.umgebung` JSON scene file
 * declaring both its initial state and its governing `PhysicsProfileConfig`.
 * Keeping Mainstream (`MainstreamGravityParams`) and Unified
 * (`UnifiedFieldParams`) parameters cleanly separated inside one schema lets
 * researchers run controlled side-by-side comparisons on identical initial
 * conditions without altering engine code.
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
        /**
         * @brief 2nd-order symplectic, time-reversible Kick-Drift-Kick Velocity Verlet.
         *
         * @details
         * Preserves phase-space volume (Liouville's theorem) and bounds total
         * mechanical energy error without secular drift over long orbital runs.
         */
        VelocityVerlet = 0,

        /**
         * @brief 1st-order explicit Euler integrator (non-symplectic; primarily
         *        useful for pedagogical error comparisons against Velocity Verlet).
         */
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
         *        Defaults to CODATA 2018/2022 SI value $6.67430 \times 10^{-11}\ \text{m}^3\,\text{kg}^{-1}\,\text{s}^{-2}$;
         *        set to `1.0` for dimensionless normalized N-body benchmarks.
         */
        double gravitationalConstant{6.67430e-11};

        /**
         * @brief Plummer gravitational softening length $\varepsilon \ge 0$ (in scene length units).
         *        Set to `0.0` for exact unsoftened analytical Kepler/choreography tests.
         */
        double softeningLength{0.0};
    };

    /**
     * @struct UnifiedFieldParams
     * @brief Forward-compatible schema parameters for Pillar 2 (`models/unified/`)
     *        synthesizing `submodules/research/Thinkers/` concepts.
     *
     * @details
     * Stored cleanly in `.umgebung` scene files so identical initial body states
     * can switch between `Mainstream` and `Unified` profiles. Pure `Mainstream`
     * solvers ignore these fields entirely.
     */
    struct UnifiedFieldParams {
        /** @brief Vacuum Planck spherical unit / harmonic vacuum density parameter (Haramein). */
        double harmonicVacuumDensity{0.0};

        /** @brief Love binding-medium frequency/coupling coefficient (Danaan / Unified). */
        double loveBindingCoupling{0.0};

        /** @brief Phryll life-force / crystal vortex ambient field amplitude (Danaan / Willis / Vogel). */
        double phryllFieldAmplitude{0.0};

        /** @brief Baseline vibrational density octave for the universe (default `3.0` = 3rd density). */
        double baseVibrationalDensity{3.0};

        /** @brief Enables non-linear spherical/gravity-coupled local time fields when true. */
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
     *
     * @param text   String value read from JSON (`"Mainstream"`, `"Unified"`, `"Custom"`).
     * @param outVal Populated enum value on success.
     * @return `true` if `text` matched a known profile; `false` otherwise.
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
