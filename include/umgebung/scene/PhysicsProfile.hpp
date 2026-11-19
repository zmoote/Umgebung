/**
 * @file PhysicsProfile.hpp
 * @brief Declarative simulation scope, physical `SimulationDomain`, multi-scale
 *        tier, composable three-tier Model Registry identifiers, and per-domain /
 *        per-model parameter blocks for `.umgebung` Scenarios, Bubble Universes,
 *        and Flower of Life Multiverses.
 *
 * @details
 * Implements the `SimulationDomain` and three-tier model architecture documented
 * in `GEMINI.md`:
 *   - **Explicit Simulation Domains (`SimulationDomain`):**
 *       - `SimulationDomain::NBodyDynamics`         — Gravitational / orbital SoA dynamics
 *       - `SimulationDomain::FieldGrid2D`           — 2D CUDA spatial EM / Cavity / Torsion / Vacuum field grid
 *       - `SimulationDomain::RadialAndScaleProfile` — 1D radial horizon or logarithmic multi-scale analytical sweep
 *   - **Tier 1 — Mainstream Physics Baselines (`models/mainstream/`):**
 *       - `ModelId::MainstreamNewtonian`     (`"Mainstream::Newtonian"`)
 *       - `ModelId::MainstreamDarkMatterNFW` (`"Mainstream::DarkMatterNFW"`)
 *       - `ModelId::MainstreamMaxwellEM`     (`"Mainstream::MaxwellEM"`)
 *       - `ModelId::MainstreamSchwarzschildGR` (`"Mainstream::SchwarzschildGR"`)
 *   - **Tier 2 — Standalone Verbatim Thinker Modules (`models/thinkers/`):**
 *       - `ModelId::ThinkerEssonne`          (`"Thinker::Essonne"`)
 *       - `ModelId::ThinkerHaramein`         (`"Thinker::Haramein"`)
 *       - `ModelId::ThinkerPais`             (`"Thinker::Pais"`)
 *       - `ModelId::ThinkerDanaan`           (`"Thinker::Danaan"`)
 *       - `ModelId::ThinkerVogelWillis`      (`"Thinker::VogelWillis"`)
 *   - **Tier 3 — Combined Unified Synthesis (`models/unified/`):**
 *       - `ModelId::UnifiedSynthesis`        (`"Unified::Synthesis"`)
 */

#pragma once

#include <cstdint>
#include <string>

namespace umgebung::scene {

    /**
     * @enum SimulationScope
     * @brief Explicit topological scope distinguishing standalone Scenarios from
     *        multi-scale Bubble Universes and Elena Danaan's Flower of Life Multiverse.
     */
    enum class SimulationScope : std::uint8_t {
        /** @brief Standalone computational physics experiment, device test, or benchmark. */
        Scenario = 0,
        /** @brief Multi-scale Bubble Universe delegating sub-scales to domain-appropriate sub-simulations. */
        BubbleUniverse = 1,
        /** @brief Elena Danaan / Unified Flower of Life geometric lattice of intersecting Bubble Universes & the Void. */
        FlowerOfLifeMultiverse = 2
    };

    /**
     * @enum SimulationDomain
     * @brief Explicit physical domain governing a Scenario or Bubble Universe
     *        tier so non-orbital problems are never forced into N-body orbits.
     */
    enum class SimulationDomain : std::uint8_t {
        /** @brief Gravitational / orbital particle SoA dynamics $(\mathbf{r}_i, \mathbf{v}_i, m_i)$. */
        NBodyDynamics = 0,
        /** @brief 2D CUDA spatial field grid (Maxwell FDTD EM, Pais HEEMFG cavity, Vogel/Willis 4096 Hz crystal, Essonne vacuum). */
        FieldGrid2D = 1,
        /** @brief 1D radial horizon or logarithmic multi-scale analytical sweep (Schwarzschild GR, Haramein PSU, Essonne horizon). */
        RadialAndScaleProfile = 2
    };

    /** @brief Bitmask type encoding the set of `SimulationDomain`s a model physically supports. */
    using SimulationDomainMask = std::uint32_t;

    /** @brief Bitmask flag for `SimulationDomain::NBodyDynamics`. */
    inline constexpr SimulationDomainMask kDomainNBodyDynamics = (1U << 0U);

    /** @brief Bitmask flag for `SimulationDomain::FieldGrid2D`. */
    inline constexpr SimulationDomainMask kDomainFieldGrid2D = (1U << 1U);

    /** @brief Bitmask flag for `SimulationDomain::RadialAndScaleProfile`. */
    inline constexpr SimulationDomainMask kDomainRadialAndScaleProfile = (1U << 2U);

    /** @brief Converts a `SimulationDomain` enumerator to its single-bit `SimulationDomainMask`. */
    [[nodiscard]] constexpr SimulationDomainMask DomainToMask(SimulationDomain domain) noexcept {
        switch (domain) {
            case SimulationDomain::NBodyDynamics:         return kDomainNBodyDynamics;
            case SimulationDomain::FieldGrid2D:           return kDomainFieldGrid2D;
            case SimulationDomain::RadialAndScaleProfile: return kDomainRadialAndScaleProfile;
        }
        return kDomainNBodyDynamics;
    }

    /** @brief Returns `true` when `mask` includes `domain`. */
    [[nodiscard]] constexpr bool MaskSupportsDomain(
        SimulationDomainMask mask,
        SimulationDomain domain
    ) noexcept {
        return (mask & DomainToMask(domain)) != 0U;
    }

    /**
     * @enum ScaleTier
     * @brief Multi-scale ontological hierarchy level supporting cosmological-to-subatomic zoom.
     */
    enum class ScaleTier : std::uint8_t {
        Multiverse = 0,
        BubbleUniverse = 1,
        Galaxy = 2,
        StarSystem = 3,
        Planet = 4,
        EntityOrDevice = 5,
        Atom = 6,
        SubatomicPSU = 7
    };

    /**
     * @enum PhysicsProfileType
     * @brief Broad theoretical category retained for schema clarity and backwards compatibility.
     */
    enum class PhysicsProfileType : std::uint8_t {
        Mainstream = 0,
        Unified = 1,
        Custom = 2
    };

    /**
     * @enum ModelId
     * @brief Canonical identifier for every concrete solver registered in
     *        `umgebung::models::ModelRegistry`.
     */
    enum class ModelId : std::uint8_t {
        // Tier 1: Mainstream Physics Baselines
        MainstreamNewtonian = 0,
        MainstreamDarkMatterNFW = 1,
        MainstreamMaxwellEM = 8,
        MainstreamSchwarzschildGR = 9,

        // Tier 2: Standalone Verbatim Thinker Modules
        ThinkerEssonne = 2,
        ThinkerHaramein = 3,
        ThinkerPais = 4,
        ThinkerDanaan = 5,
        ThinkerVogelWillis = 6,

        // Tier 3: Combined Unified Synthesis
        UnifiedSynthesis = 7
    };

    /**
     * @enum IntegratorType
     * @brief Numerical time-integration scheme selected for dynamical evolution.
     */
    enum class IntegratorType : std::uint8_t {
        /** @brief 2nd-order symplectic, time-reversible Kick-Drift-Kick Velocity Verlet (or leapfrog Yee FDTD). */
        VelocityVerlet = 0,
        /** @brief 1st-order explicit Euler integrator (non-symplectic pedagogical baseline). */
        EulerExplicit = 1
    };

    /**
     * @struct FieldGrid2DConfig
     * @brief Configuration for `SimulationDomain::FieldGrid2D` GPU spatial field
     *        solvers (2D TMz FDTD EM cavities, Pais HEEMFG, Vogel/Willis 4096 Hz
     *        Phryll crystal cavity, and Essonne 2D vacuum pressure fields).
     */
    struct FieldGrid2DConfig {
        /** @brief Number of grid cells along the X axis (default `128`). */
        std::uint32_t gridWidth{128U};

        /** @brief Number of grid cells along the Y axis (default `128`). */
        std::uint32_t gridHeight{128U};

        /** @brief Physical half-extent $L > 0$ so coordinates span $[-L, +L] \times [-L, +L]$. */
        double domainHalfExtent{2.0};

        /** @brief Characteristic wave/propagation speed $c_{\text{grid}} > 0$ in grid units (governs 2D CFL bound). */
        double waveSpeed{1.0};

        /** @brief Normalized cavity excitation frequency $\omega_{\text{drive}} / (2\pi)$ in grid time units. */
        double sourceFrequency{1.5};

        /** @brief Mild boundary sponge / cavity dissipation factor $\gamma \ge 0$. */
        double dampingCoeff{0.002};

        /** @brief Human-readable label describing the visualized primary field quantity. */
        std::string fieldQuantityLabel{"|S| / Field Amplitude"};
    };

    /**
     * @struct RadialScaleConfig
     * @brief Configuration for `SimulationDomain::RadialAndScaleProfile` 1D radial
     *        horizon or logarithmic multi-scale analytical sweeps.
     */
    struct RadialScaleConfig {
        /** @brief Number of evaluated sample points along the sweep axis (default `256`). */
        std::uint32_t sampleCount{256U};

        /** @brief Minimum sweep coordinate (radius $r_{\min}$ in meters or $r/R_s$). */
        double rangeMin{0.05};

        /** @brief Maximum sweep coordinate (radius $r_{\max}$ in meters or $r/R_s$). */
        double rangeMax{5.0};

        /** @brief When `true`, sample points are logarithmically spaced across `[rangeMin, rangeMax]`. */
        bool logarithmicScale{false};

        /** @brief Reference Schwarzschild radius $R_s > 0$ for horizon / vacuum-pressure sweeps. */
        double referenceSchwarzschildRadius{1.0};

        /** @brief Reference homogeneous body radius $R > 0$ (e.g., `1.5 * R_s` for Essonne `8052.pdf` Eq. 39). */
        double referenceBodyRadius{1.5};

        /** @brief Sweep family selector: `"HolographicMassScale"` or `"VacuumHorizon"`. */
        std::string sweepKind{"VacuumHorizon"};

        /** @brief Horizontal axis label for plotter and JSON telemetry. */
        std::string xAxisLabel{"r / R_s"};

        /** @brief Vertical axis label for plotter and JSON telemetry. */
        std::string yAxisLabel{"P(r)/P_0 and c'(r)/c"};
    };

    /**
     * @struct MainstreamGravityParams
     * @brief Governing parameters for textbook Newtonian pairwise N-body gravity.
     */
    struct MainstreamGravityParams {
        double gravitationalConstant{6.67430e-11};
        double softeningLength{0.0};
    };

    /**
     * @struct DarkMatterNFWParams
     * @brief Navarro-Frenk-White (NFW, 1996/1997) spherical dark matter halo baseline parameters.
     */
    struct DarkMatterNFWParams {
        double characteristicDensityRho0{0.0};
        double scaleRadiusRs{2.0};
    };

    /**
     * @struct HarameinParams
     * @brief Verbatim parameters for `Thinker::Haramein` (`Haramein342013PRRI3363.pdf` & spacetime torque).
     */
    struct HarameinParams {
        double harmonicVacuumDensity{5.155e96};
        double spacetimeTorqueOmegaZ{0.0};
    };

    /**
     * @struct PaisHeemfgParams
     * @brief Verbatim parameters for `Thinker::Pais` (`US10144532.pdf` Eq. 1–2 HEEMFG cavity).
     */
    struct PaisHeemfgParams {
        double geometricFactor{1.0};
        double surfaceChargeDensity{0.0};
        double rotationRadius{2.0};
        double angularFrequencyRadS{0.0};
        double vibrationAmplitude{0.0};
        double vibrationFrequencyHz{0.0};
        double translationSpeed{0.0};
        double inertialReductionCoupling{0.0};
    };

    /**
     * @struct VogelWillisCrystalParams
     * @brief Verbatim parameters for `Thinker::VogelWillis` (`PhryllGenInstructions.pdf`).
     */
    struct VogelWillisCrystalParams {
        double torsionCoilFrequencyHz{4096.0};
        double driveVoltageV{20.0};
        double baseTerminationAngleDeg{51.84};
        double topTerminationAngleDeg{60.0};
        double saturationThreshold{1000.0};
        double phryllCoupling{0.0};
    };

    /**
     * @struct UnifiedFieldParams
     * @brief Composable parameter bag carrying Essonne, Haramein, Pais, Danaan,
     *        and Vogel/Willis parameters cleanly within `.umgebung` files.
     */
    struct UnifiedFieldParams {
        double harmonicVacuumDensity{5.155e96};
        HarameinParams haramein{};

        double referenceLightSpeed{299792458.0};
        bool aetherVortexEnabled{false};
        double aetherVortexCriticalRadius{1.0};
        double aetherVortexScaleHeight{1.0};
        double aetherVortexVerticalFieldGz{0.0};

        PaisHeemfgParams pais{};

        double loveBindingCoupling{0.0};
        double phryllFieldAmplitude{0.0};
        double baseVibrationalDensity{3.0};
        bool nonLinearTimeEnabled{false};

        VogelWillisCrystalParams vogelWillis{};
    };

    /**
     * @struct PhysicsProfileConfig
     * @brief Complete declarative physics configuration embedded in a Scenario or Bubble Universe.
     */
    struct PhysicsProfileConfig {
        PhysicsProfileType type{PhysicsProfileType::Mainstream};
        ModelId defaultModel{ModelId::MainstreamNewtonian};
        std::string recommendedCompareModel;
        IntegratorType integrator{IntegratorType::VelocityVerlet};
        std::string unitSystem{"Normalized"};
        double timestep{1.0e-3};
        std::uint64_t defaultSteps{1000ULL};

        MainstreamGravityParams mainstreamGravity{};
        DarkMatterNFWParams darkMatterNFW{};
        FieldGrid2DConfig fieldGrid2D{};
        RadialScaleConfig radialScale{};
        UnifiedFieldParams unifiedField{};
    };

    // =========================================================================
    // String <-> Enum Serialization Helpers
    // =========================================================================

    [[nodiscard]] inline const char* SimulationScopeToString(SimulationScope scope) noexcept {
        switch (scope) {
            case SimulationScope::Scenario:               return "Scenario";
            case SimulationScope::BubbleUniverse:         return "BubbleUniverse";
            case SimulationScope::FlowerOfLifeMultiverse: return "FlowerOfLifeMultiverse";
        }
        return "Scenario";
    }

    [[nodiscard]] inline bool SimulationScopeFromString(const std::string& text, SimulationScope& outVal) noexcept {
        if (text == "Scenario" || text == "scenario") { outVal = SimulationScope::Scenario; return true; }
        if (text == "BubbleUniverse" || text == "Universe" || text == "universe" || text == "bubble_universe") {
            outVal = SimulationScope::BubbleUniverse; return true;
        }
        if (text == "FlowerOfLifeMultiverse" || text == "Multiverse" || text == "multiverse") {
            outVal = SimulationScope::FlowerOfLifeMultiverse; return true;
        }
        return false;
    }

    [[nodiscard]] inline const char* SimulationDomainToString(SimulationDomain domain) noexcept {
        switch (domain) {
            case SimulationDomain::NBodyDynamics:         return "NBodyDynamics";
            case SimulationDomain::FieldGrid2D:           return "FieldGrid2D";
            case SimulationDomain::RadialAndScaleProfile: return "RadialAndScaleProfile";
        }
        return "NBodyDynamics";
    }

    [[nodiscard]] inline bool SimulationDomainFromString(const std::string& text, SimulationDomain& outVal) noexcept {
        if (text == "NBodyDynamics" || text == "nbody" || text == "Orbital") {
            outVal = SimulationDomain::NBodyDynamics; return true;
        }
        if (text == "FieldGrid2D" || text == "field_grid_2d" || text == "EMField" || text == "CavityField") {
            outVal = SimulationDomain::FieldGrid2D; return true;
        }
        if (text == "RadialAndScaleProfile" || text == "radial_scale" || text == "ScaleSweep") {
            outVal = SimulationDomain::RadialAndScaleProfile; return true;
        }
        return false;
    }

    [[nodiscard]] inline const char* ScaleTierToString(ScaleTier tier) noexcept {
        switch (tier) {
            case ScaleTier::Multiverse:     return "Multiverse";
            case ScaleTier::BubbleUniverse: return "BubbleUniverse";
            case ScaleTier::Galaxy:         return "Galaxy";
            case ScaleTier::StarSystem:     return "StarSystem";
            case ScaleTier::Planet:         return "Planet";
            case ScaleTier::EntityOrDevice: return "EntityOrDevice";
            case ScaleTier::Atom:           return "Atom";
            case ScaleTier::SubatomicPSU:   return "SubatomicPSU";
        }
        return "StarSystem";
    }

    [[nodiscard]] inline bool ScaleTierFromString(const std::string& text, ScaleTier& outVal) noexcept {
        if (text == "Multiverse")     { outVal = ScaleTier::Multiverse; return true; }
        if (text == "BubbleUniverse") { outVal = ScaleTier::BubbleUniverse; return true; }
        if (text == "Galaxy")         { outVal = ScaleTier::Galaxy; return true; }
        if (text == "StarSystem")     { outVal = ScaleTier::StarSystem; return true; }
        if (text == "Planet")         { outVal = ScaleTier::Planet; return true; }
        if (text == "EntityOrDevice") { outVal = ScaleTier::EntityOrDevice; return true; }
        if (text == "Atom")           { outVal = ScaleTier::Atom; return true; }
        if (text == "SubatomicPSU")   { outVal = ScaleTier::SubatomicPSU; return true; }
        return false;
    }

    [[nodiscard]] inline const char* ModelIdToString(ModelId id) noexcept {
        switch (id) {
            case ModelId::MainstreamNewtonian:       return "Mainstream::Newtonian";
            case ModelId::MainstreamDarkMatterNFW:   return "Mainstream::DarkMatterNFW";
            case ModelId::MainstreamMaxwellEM:       return "Mainstream::MaxwellEM";
            case ModelId::MainstreamSchwarzschildGR: return "Mainstream::SchwarzschildGR";
            case ModelId::ThinkerEssonne:            return "Thinker::Essonne";
            case ModelId::ThinkerHaramein:           return "Thinker::Haramein";
            case ModelId::ThinkerPais:               return "Thinker::Pais";
            case ModelId::ThinkerDanaan:             return "Thinker::Danaan";
            case ModelId::ThinkerVogelWillis:        return "Thinker::VogelWillis";
            case ModelId::UnifiedSynthesis:          return "Unified::Synthesis";
        }
        return "Mainstream::Newtonian";
    }

    [[nodiscard]] inline bool ModelIdFromString(const std::string& text, ModelId& outVal) noexcept {
        if (text == "Mainstream::Newtonian" || text == "newtonian" || text == "Mainstream") {
            outVal = ModelId::MainstreamNewtonian; return true;
        }
        if (text == "Mainstream::DarkMatterNFW" || text == "nfw" || text == "DarkMatterNFW") {
            outVal = ModelId::MainstreamDarkMatterNFW; return true;
        }
        if (text == "Mainstream::MaxwellEM" || text == "MaxwellEM" || text == "maxwell") {
            outVal = ModelId::MainstreamMaxwellEM; return true;
        }
        if (text == "Mainstream::SchwarzschildGR" || text == "SchwarzschildGR" || text == "schwarzschild") {
            outVal = ModelId::MainstreamSchwarzschildGR; return true;
        }
        if (text == "Thinker::Essonne" || text == "Essonne" || text == "essonne") {
            outVal = ModelId::ThinkerEssonne; return true;
        }
        if (text == "Thinker::Haramein" || text == "Haramein" || text == "haramein") {
            outVal = ModelId::ThinkerHaramein; return true;
        }
        if (text == "Thinker::Pais" || text == "Pais" || text == "pais") {
            outVal = ModelId::ThinkerPais; return true;
        }
        if (text == "Thinker::Danaan" || text == "Danaan" || text == "danaan") {
            outVal = ModelId::ThinkerDanaan; return true;
        }
        if (text == "Thinker::VogelWillis" || text == "VogelWillis" || text == "vogel_willis" || text == "Phryll") {
            outVal = ModelId::ThinkerVogelWillis; return true;
        }
        if (text == "Unified::Synthesis" || text == "Unified" || text == "unified") {
            outVal = ModelId::UnifiedSynthesis; return true;
        }
        return false;
    }

    [[nodiscard]] inline const char* PhysicsProfileTypeToString(PhysicsProfileType type) noexcept {
        switch (type) {
            case PhysicsProfileType::Mainstream: return "Mainstream";
            case PhysicsProfileType::Unified:    return "Unified";
            case PhysicsProfileType::Custom:     return "Custom";
        }
        return "Mainstream";
    }

    [[nodiscard]] inline bool PhysicsProfileTypeFromString(
        const std::string& text,
        PhysicsProfileType& outVal
    ) noexcept {
        if (text == "Mainstream" || text == "mainstream") { outVal = PhysicsProfileType::Mainstream; return true; }
        if (text == "Unified" || text == "unified")       { outVal = PhysicsProfileType::Unified; return true; }
        if (text == "Custom" || text == "custom")         { outVal = PhysicsProfileType::Custom; return true; }
        return false;
    }

    [[nodiscard]] inline const char* IntegratorTypeToString(IntegratorType integrator) noexcept {
        switch (integrator) {
            case IntegratorType::VelocityVerlet: return "VelocityVerlet";
            case IntegratorType::EulerExplicit:  return "EulerExplicit";
        }
        return "VelocityVerlet";
    }

    [[nodiscard]] inline bool IntegratorTypeFromString(
        const std::string& text,
        IntegratorType& outVal
    ) noexcept {
        if (text == "VelocityVerlet" || text == "velocity_verlet") { outVal = IntegratorType::VelocityVerlet; return true; }
        if (text == "EulerExplicit" || text == "euler_explicit")   { outVal = IntegratorType::EulerExplicit; return true; }
        return false;
    }

} // namespace umgebung::scene
