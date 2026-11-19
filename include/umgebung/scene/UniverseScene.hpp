/**
 * @file UniverseScene.hpp
 * @brief Shared core Entity/Field scene schema supporting standalone
 *        **Scenarios** (`*.scenario.umgebung`) and multi-scale **Bubble
 *        Universes** (`*.umgebung`) across all explicit `SimulationDomain`s
 *        (`NBodyDynamics`, `FieldGrid2D`, and `RadialAndScaleProfile`).
 *
 * @details
 * Prevents non-orbital problems from being forced into nonsensical N-body
 * orbits:
 *   - Every scene declares its governing `domain` (`SimulationDomain`).
 *   - Multi-scale Bubble Universes keep macroscopic gravitating bodies in
 *     `bodies` while delegating sub-scale devices, atoms, and subatomic PSU
 *     probes to `subSimulations` (`ScaleTierSubSimulation`), each pointing to
 *     its own domain-appropriate `.scenario.umgebung` file!
 *
 * @note Pure C++20 + `nlohmann_json`; zero CUDA, Vulkan, GLFW, or ImGui deps.
 */

#pragma once

#include "umgebung/scene/PhysicsProfile.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace umgebung::scene {

    /**
     * @struct Vec3d
     * @brief IEEE-754 double-precision 3D Cartesian vector `(x, y, z)`.
     */
    struct Vec3d {
        double x{0.0};
        double y{0.0};
        double z{0.0};

        [[nodiscard]] constexpr Vec3d operator+(const Vec3d& rhs) const noexcept {
            return {x + rhs.x, y + rhs.y, z + rhs.z};
        }

        [[nodiscard]] constexpr Vec3d operator-(const Vec3d& rhs) const noexcept {
            return {x - rhs.x, y - rhs.y, z - rhs.z};
        }

        [[nodiscard]] constexpr Vec3d operator*(double scalar) const noexcept {
            return {x * scalar, y * scalar, z * scalar};
        }

        [[nodiscard]] static constexpr double Dot(const Vec3d& a, const Vec3d& b) noexcept {
            return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
        }

        [[nodiscard]] static constexpr Vec3d Cross(const Vec3d& a, const Vec3d& b) noexcept {
            return {
                (a.y * b.z) - (a.z * b.y),
                (a.z * b.x) - (a.x * b.z),
                (a.x * b.y) - (a.y * b.x)
            };
        }

        [[nodiscard]] constexpr double NormSquared() const noexcept {
            return Dot(*this, *this);
        }

        [[nodiscard]] double Norm() const noexcept {
            return std::sqrt(NormSquared());
        }
    };

    /**
     * @struct BodyEntity
     * @brief Persistent descriptor for a gravitating body (in `NBodyDynamics`)
     *        or stationary cavity/crystal/coil anchor (in `FieldGrid2D`).
     */
    struct BodyEntity {
        std::string id;
        std::string name;
        std::string category{"Body"};
        ScaleTier scaleTier{ScaleTier::StarSystem};
        std::string parentId;
        double mass{1.0};
        double radius{0.05};
        Vec3d position{};
        Vec3d velocity{};

        // Thinker / Unified Entity Properties
        double vibrationalDensity{3.0};
        double consciousnessCoherence{0.0};
        double phryllCharge{0.0};
    };

    /**
     * @struct ScaleTierSubSimulation
     * @brief Multi-scale link attached to a `BubbleUniverse` delegating a
     *        lower-scale tier (e.g., an `EntityOrDevice` Vogel Phryll crystal
     *        or a `SubatomicPSU` proton) to its own domain-appropriate
     *        `.scenario.umgebung` simulation rather than mixing it into the
     *        macroscopic N-body gravitational array.
     */
    struct ScaleTierSubSimulation {
        /** @brief Unique identifier for this sub-simulation link. */
        std::string id;

        /** @brief Human-readable label shown in the Bubble Universe hierarchy tree. */
        std::string name;

        /** @brief Multi-scale tier represented by this sub-simulation (`EntityOrDevice`, `Atom`, `SubatomicPSU`, etc.). */
        ScaleTier scaleTier{ScaleTier::EntityOrDevice};

        /** @brief `SimulationDomain` of the linked scenario (`FieldGrid2D`, `RadialAndScaleProfile`, etc.). */
        SimulationDomain domain{SimulationDomain::FieldGrid2D};

        /** @brief Optional `id` of the parent body or parent sub-simulation in the Bubble Universe tree. */
        std::string parentId;

        /** @brief Relative or resolved path to the domain-appropriate `.scenario.umgebung` file. */
        std::filesystem::path scenarioPath;
    };

    /**
     * @struct UniverseScene
     * @brief Canonical in-memory representation of a `.scenario.umgebung`
     *        Scenario or `.umgebung` Bubble Universe file across all domains.
     */
    struct UniverseScene {
        static constexpr std::uint32_t kCurrentSchemaVersion = 1U;

        std::uint32_t schemaVersion{kCurrentSchemaVersion};
        SimulationScope scope{SimulationScope::Scenario};

        /** @brief Explicit governing physical domain (`NBodyDynamics`, `FieldGrid2D`, or `RadialAndScaleProfile`). */
        SimulationDomain domain{SimulationDomain::NBodyDynamics};

        std::string universeId;
        std::string name;
        std::string description;
        std::filesystem::path sourcePath;
        PhysicsProfileConfig physicsProfile{};

        /** @brief Gravitating bodies (`NBodyDynamics`) or stationary device geometry (`FieldGrid2D`). */
        std::vector<BodyEntity> bodies;

        /** @brief Domain-appropriate sub-scale simulations linked inside a multi-scale `BubbleUniverse`. */
        std::vector<ScaleTierSubSimulation> subSimulations;

        [[nodiscard]] bool Validate(std::string& outError) const;
        [[nodiscard]] nlohmann::json ToJson() const;

        [[nodiscard]] static bool FromJson(
            const nlohmann::json& jsonObj,
            UniverseScene& outScene,
            std::string& outError
        );

        [[nodiscard]] static bool LoadFromFile(
            const std::filesystem::path& filePath,
            UniverseScene& outScene,
            std::string& outError
        );

        [[nodiscard]] bool SaveToFile(
            const std::filesystem::path& filePath,
            std::string& outError
        ) const;
    };

    using SimulationScene = UniverseScene;

} // namespace umgebung::scene
