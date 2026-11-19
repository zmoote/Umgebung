/**
 * @file PhysicsProfile.hpp
 * @brief Declarative simulation scope, multi-scale tier, composable three-tier
 *        Model Registry identifiers, and per-model parameter blocks for `.umgebung`
 *        Scenarios, Bubble Universes, and Flower of Life Multiverses.
 *
 * @details
 * Implements the updated three-tier model hierarchy documented in `GEMINI.md`:
 *   - **Tier 1 — Mainstream Physics Baselines (`models/mainstream/`):**
 *       - `ModelId::MainstreamNewtonian`     (`"Mainstream::Newtonian"`)
 *       - `ModelId::MainstreamDarkMatterNFW` (`"Mainstream::DarkMatterNFW"`)
 *   - **Tier 2 — Standalone Verbatim Thinker Modules (`models/thinkers/`):**
 *       - `ModelId::ThinkerEssonne`          (`"Thinker::Essonne"`)
 *       - `ModelId::ThinkerHaramein`         (`"Thinker::Haramein"`)
 *       - `ModelId::ThinkerPais`             (`"Thinker::Pais"`)
 *       - `ModelId::ThinkerDanaan`           (`"Thinker::Danaan"`)
 *       - `ModelId::ThinkerVogelWillis`      (`"Thinker::VogelWillis"`)
 *   - **Tier 3 — Combined Unified Synthesis (`models/unified/`):**
 *       - `ModelId::UnifiedSynthesis`        (`"Unified::Synthesis"`)
 *
 * Also defines the three explicit topological simulation scopes:
 *   - `SimulationScope::Scenario`               (`*.scenario.umgebung`)
 *   - `SimulationScope::BubbleUniverse`         (`*.umgebung`)
 *   - `SimulationScope::FlowerOfLifeMultiverse` (`*.multiverse.umgebung`)
 */

#pragma once

#include <cstdint>
#include <string>

namespace umgebung::scene {

    /**
     * @enum SimulationScope
     * @brief Explicit topological scope distinguishing standalone physics
     *        experiments from Bubble Universes and Elena Danaan's Flower of
     *        Life Multiverse.
     */
    enum class SimulationScope : std::uint8_t {
        /** @brief Standalone physics experiment, device test, or benchmark (no Multiverse wrapper). */
        Scenario = 0,
        /** @brief Multi-scale Bubble Universe (`Universe -> Galaxy -> Star -> Planet -> Entity -> Atom -> PSU`). */
        BubbleUniverse = 1,
        /** @brief Elena Danaan / Unified Flower of Life geometric lattice of intersecting Bubble Universes & the Void. */
        FlowerOfLifeMultiverse = 2
    };

    /**
     * @enum ScaleTier
     * @brief Multi-scale ontological hierarchy level attached to entities and
     *        universes to support continuous cosmological-to-subatomic zoom.
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
        /** @brief Tier 1 verified textbook physics baselines. */
        Mainstream = 0,
        /** @brief Tier 3 full Unified Synthesis composing all Thinker modules. */
        Unified = 1,
        /** @brief Tier 2 standalone Thinker module or custom A/B composition. */
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
        /** @brief 2nd-order symplectic, time-reversible Kick-Drift-Kick Velocity Verlet. */
        VelocityVerlet = 0,
        /** @brief 1st-order explicit Euler integrator (non-symplectic pedagogical baseline). */
        EulerExplicit = 1
    };

    /**
     * @struct MainstreamGravityParams
     * @brief Governing parameters for textbook Newtonian pairwise N-body gravity.
     */
    struct MainstreamGravityParams {
        /** @brief Gravitational constant $G$ (SI default $6.67430 \times 10^{-11}$; `1.0` for normalized units). */
        double gravitationalConstant{6.67430e-11};

        /** @brief Plummer gravitational softening length $\varepsilon \ge 0$. */
        double softeningLength{0.0};
    };

    /**
     * @struct DarkMatterNFWParams
     * @brief Navarro-Frenk-White (NFW, 1996/1997) spherical dark matter halo
     *        baseline parameters for `Mainstream::DarkMatterNFW`.
     *
     * @details
     * Density profile: $\rho_{\text{NFW}}(r) = \frac{\rho_0}{(r / r_s)(1 + r / r_s)^2}$
     * Enclosed mass:   $M_{\text{NFW}}(r) = 4\pi \rho_0 r_s^3 \left[\ln(1 + r/r_s) - \frac{r/r_s}{1 + r/r_s}\right]$
     * Potential:       $\Phi_{\text{NFW}}(r) = -4\pi G \rho_0 r_s^3 \frac{\ln(1 + r/r_s)}{r}$
     */
    struct DarkMatterNFWParams {
        /** @brief Characteristic halo density scale $\rho_0 \ge 0$ (scene mass / length$^3$). */
        double characteristicDensityRho0{0.0};

        /** @brief Halo scale radius $r_s > 0$ (scene length units). */
        double scaleRadiusRs{2.0};
    };

    /**
     * @struct HarameinParams
     * @brief Verbatim parameters for `Thinker::Haramein` (`Haramein342013PRRI3363.pdf`
     *        & `Scale-Unification` spacetime torque/Coriolis extension).
     */
    struct HarameinParams {
        /** @brief Ambient vacuum Planck energy density $\rho_{\text{vac}}$ ($\text{kg}/\text{m}^3$). */
        double harmonicVacuumDensity{5.155e96};

        /**
         * @brief Haramein-Rauscher spacetime-torque background rotation rate
         *        $\Omega_z$ ($\text{rad}/\text{time}$) generating Coriolis force
         *        $\mathbf{a}_{\text{Cor}} = -2 \boldsymbol{\Omega} \times \mathbf{v}$.
         */
        double spacetimeTorqueOmegaZ{0.0};
    };

    /**
     * @struct PaisHeemfgParams
     * @brief Verbatim parameters for `Thinker::Pais` (`US10144532.pdf` Eq. 1–2):
     *        High-Energy Electromagnetic Field Generator (HEEMFG) inertial mass reduction.
     */
    struct PaisHeemfgParams {
        /** @brief Geometric shape factor $f_G$ (`1.0` for disc configuration). */
        double geometricFactor{1.0};

        /** @brief Surface charge density $\sigma$ ($\text{C}/\text{m}^2$). */
        double surfaceChargeDensity{0.0};

        /** @brief Disc axial rotation radius $R_r$ ($\text{m}$). */
        double rotationRadius{2.0};

        /** @brief Axial rotation angular frequency $\omega$ ($\text{rad}/\text{s}$). */
        double angularFrequencyRadS{0.0};

        /** @brief Harmonic vibration amplitude $R_v$ ($\text{m}$). */
        double vibrationAmplitude{0.0};

        /** @brief Harmonic vibration frequency $\nu$ ($\text{Hz}$). */
        double vibrationFrequencyHz{0.0};

        /** @brief Curvilinear translation speed $v_R$ ($\text{m}/\text{s}$). */
        double translationSpeed{0.0};

        /**
         * @brief Vacuum-polarization inertial-mass-reduction coupling $\kappa_{\text{Pais}}$
         *        mapping $S_{\max}$ to $m_{\text{eff}} = m_0 / (1 + \kappa_{\text{Pais}} S_{\max})$.
         */
        double inertialReductionCoupling{0.0};
    };

    /**
     * @struct VogelWillisCrystalParams
     * @brief Verbatim parameters for `Thinker::VogelWillis` (`PhryllGenInstructions.pdf`):
     *        Vogel/Eredyon crystal geometry, 4096 Hz orthogonal copper/silver torsion coils,
     *        Phryll generation, and saturation/singing threshold monitoring.
     */
    struct VogelWillisCrystalParams {
        /** @brief Alternating `180°` phase-shifted coil pulse frequency (nominal `4096.0 Hz`). */
        double torsionCoilFrequencyHz{4096.0};

        /** @brief Coil excitation amplitude in Volts (`20.0 V` nominal safe, `60.0 V` upper limit). */
        double driveVoltageV{20.0};

        /** @brief Crystal base termination cut angle in degrees (nominal `51.84°`). */
        double baseTerminationAngleDeg{51.84};

        /** @brief Crystal upper termination cut angle in degrees (nominal `60.0°`). */
        double topTerminationAngleDeg{60.0};

        /** @brief Cumulative energy/mass saturation ceiling beyond which the crystal "sings" / implodes. */
        double saturationThreshold{1000.0};

        /** @brief Ambient Phryll field coupling constant for inter-entity regenerative/resonant forces. */
        double phryllCoupling{0.0};
    };

    /**
     * @struct UnifiedFieldParams
     * @brief Composable parameter bag carrying Essonne, Haramein, Pais, Danaan,
     *        and Vogel/Willis parameters cleanly within `.umgebung` files.
     */
    struct UnifiedFieldParams {
        // --- Haramein ---
        double harmonicVacuumDensity{5.155e96};
        HarameinParams haramein{};

        // --- Chris Essonne (`8052.pdf` & `10100.pdf`) ---
        double referenceLightSpeed{299792458.0};
        bool aetherVortexEnabled{false};
        double aetherVortexCriticalRadius{1.0};
        double aetherVortexScaleHeight{1.0};
        double aetherVortexVerticalFieldGz{0.0};

        // --- Salvatore Pais (`US10144532.pdf`) ---
        PaisHeemfgParams pais{};

        // --- Elena Danaan (Love / Vibrational Density / Non-Linear Time) ---
        double loveBindingCoupling{0.0};
        double phryllFieldAmplitude{0.0};
        double baseVibrationalDensity{3.0};
        bool nonLinearTimeEnabled{false};

        // --- Marcel Vogel / Dan Willis / Jen Han Eredyon ---
        VogelWillisCrystalParams vogelWillis{};
    };

    /**
     * @struct PhysicsProfileConfig
     * @brief Complete declarative physics configuration embedded in a Scenario or Bubble Universe.
     */
    struct PhysicsProfileConfig {
        /** @brief High-level category (`Mainstream`, `Unified`, or `Custom`). */
        PhysicsProfileType type{PhysicsProfileType::Mainstream};

        /** @brief Default concrete solver selected when this scene is loaded (overridable live via dropdown or `--model`). */
        ModelId defaultModel{ModelId::MainstreamNewtonian};

        /** @brief Optional recommended Model B for immediate A/B overlay comparison (`""` = none). */
        std::string recommendedCompareModel;

        /** @brief Selected numerical time integrator (`VelocityVerlet` default). */
        IntegratorType integrator{IntegratorType::VelocityVerlet};

        /** @brief Unit convention label (`"SI"` or `"Normalized"`). */
        std::string unitSystem{"Normalized"};

        /** @brief Integration timestep $\Delta t$ in scene time units (must be `> 0`). */
        double timestep{1.0e-3};

        /** @brief Default step count executed in headless batch mode (`--steps` overrides). */
        std::uint64_t defaultSteps{1000ULL};

        /** @brief Mainstream Newtonian pairwise gravity parameters. */
        MainstreamGravityParams mainstreamGravity{};

        /** @brief Mainstream Navarro-Frenk-White dark matter halo baseline parameters. */
        DarkMatterNFWParams darkMatterNFW{};

        /** @brief Composable parameters for all Thinker modules and `Unified::Synthesis`. */
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
        if (text == "Scenario" || text == "scenario") {
            outVal = SimulationScope::Scenario;
            return true;
        }
        if (text == "BubbleUniverse" || text == "Universe" || text == "universe" || text == "bubble_universe") {
            outVal = SimulationScope::BubbleUniverse;
            return true;
        }
        if (text == "FlowerOfLifeMultiverse" || text == "Multiverse" || text == "multiverse") {
            outVal = SimulationScope::FlowerOfLifeMultiverse;
            return true;
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
            case ModelId::MainstreamNewtonian:     return "Mainstream::Newtonian";
            case ModelId::MainstreamDarkMatterNFW: return "Mainstream::DarkMatterNFW";
            case ModelId::ThinkerEssonne:          return "Thinker::Essonne";
            case ModelId::ThinkerHaramein:         return "Thinker::Haramein";
            case ModelId::ThinkerPais:             return "Thinker::Pais";
            case ModelId::ThinkerDanaan:           return "Thinker::Danaan";
            case ModelId::ThinkerVogelWillis:      return "Thinker::VogelWillis";
            case ModelId::UnifiedSynthesis:        return "Unified::Synthesis";
        }
        return "Mainstream::Newtonian";
    }

    [[nodiscard]] inline bool ModelIdFromString(const std::string& text, ModelId& outVal) noexcept {
        if (text == "Mainstream::Newtonian" || text == "newtonian" || text == "Mainstream") {
            outVal = ModelId::MainstreamNewtonian;
            return true;
        }
        if (text == "Mainstream::DarkMatterNFW" || text == "nfw" || text == "DarkMatterNFW") {
            outVal = ModelId::MainstreamDarkMatterNFW;
            return true;
        }
        if (text == "Thinker::Essonne" || text == "Essonne" || text == "essonne") {
            outVal = ModelId::ThinkerEssonne;
            return true;
        }
        if (text == "Thinker::Haramein" || text == "Haramein" || text == "haramein") {
            outVal = ModelId::ThinkerHaramein;
            return true;
        }
        if (text == "Thinker::Pais" || text == "Pais" || text == "pais") {
            outVal = ModelId::ThinkerPais;
            return true;
        }
        if (text == "Thinker::Danaan" || text == "Danaan" || text == "danaan") {
            outVal = ModelId::ThinkerDanaan;
            return true;
        }
        if (text == "Thinker::VogelWillis" || text == "VogelWillis" || text == "vogel_willis" || text == "Phryll") {
            outVal = ModelId::ThinkerVogelWillis;
            return true;
        }
        if (text == "Unified::Synthesis" || text == "Unified" || text == "unified") {
            outVal = ModelId::UnifiedSynthesis;
            return true;
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
