/**
 * @file UniverseScene.hpp
 * @brief Shared core Entity/Field scene schema supporting both standalone
 *        **Scenarios** (`*.scenario.umgebung`) and multi-scale **Bubble
 *        Universes** (`*.umgebung`) within the Umgebung engine.
 *
 * @details
 * Both standalone computational physics experiments (`SimulationScope::Scenario`)
 * and full multi-scale Bubble Universes (`SimulationScope::BubbleUniverse`)
 * share this single canonical data representation (`UniverseScene`, with type
 * alias `SimulationScene`), eliminating duplicate serialization or solver glue
 * while allowing the UI and Multiverse container to display the exact right
 * topological hierarchy for each scope.
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
     * @brief Persistent descriptor for a single physical, electromagnetic,
     *        crystalline, atomic, or conscious entity inside a Scenario or
     *        Bubble Universe.
     */
    struct BodyEntity {
        /** @brief Unique identifier within the enclosing scene. */
        std::string id;

        /** @brief Human-readable display name shown in logs and hierarchy panels. */
        std::string name;

        /** @brief Semantic category (e.g., `"Star"`, `"Planet"`, `"HEEMFG_Craft"`, `"PhryllCrystal"`, `"Atom"`, `"PSU"`). */
        std::string category{"Body"};

        /** @brief Multi-scale zoom tier (Galaxy -> StarSystem -> Planet -> EntityOrDevice -> Atom -> SubatomicPSU). */
        ScaleTier scaleTier{ScaleTier::StarSystem};

        /** @brief Optional `id` of a parent entity within the same scene (`""` = root-level entity). */
        std::string parentId;

        /** @brief Inertial/gravitational rest mass $m > 0$ in scene mass units. */
        double mass{1.0};

        /** @brief Characteristic physical radius $R \ge 0$ in scene length units. */
        double radius{0.05};

        /** @brief Initial Cartesian position $\mathbf{r}_0 = (x, y, z)$. */
        Vec3d position{};

        /** @brief Initial Cartesian velocity $\mathbf{v}_0 = (v_x, v_y, v_z)$. */
        Vec3d velocity{};

        // ---------------------------------------------------------------------
        // Thinker / Unified Entity Properties (Ignored by Pure Mainstream Models)
        // ---------------------------------------------------------------------

        /** @brief Entity vibrational density octave (Danaan framework; default `3.0`). */
        double vibrationalDensity{3.0};

        /** @brief Consciousness coherence scalar $[0, 1]$ (Danaan / Emoto / Vogel). */
        double consciousnessCoherence{0.0};

        /** @brief Stored/emitted Phryll energy charge (Vogel / Willis / Danaan). */
        double phryllCharge{0.0};
    };

    /**
     * @struct UniverseScene
     * @brief Canonical in-memory representation of a `.scenario.umgebung`
     *        Scenario or `.umgebung` Bubble Universe file.
     */
    struct UniverseScene {
        /** @brief Current `.umgebung` JSON schema version supported by this engine. */
        static constexpr std::uint32_t kCurrentSchemaVersion = 1U;

        /** @brief Schema version parsed from JSON (must equal `kCurrentSchemaVersion`). */
        std::uint32_t schemaVersion{kCurrentSchemaVersion};

        /** @brief Topological scope (`Scenario` vs. `BubbleUniverse`). */
        SimulationScope scope{SimulationScope::Scenario};

        /** @brief Stable machine-friendly identifier (e.g., `"galaxy_periphery"`, `"sol_prime"`). */
        std::string universeId;

        /** @brief Human-readable title displayed in CLI logs and UI tabs. */
        std::string name;

        /** @brief Scientific or cosmological description of the scene. */
        std::string description;

        /** @brief Filesystem path from which this scene was loaded (empty for programmatic scenes). */
        std::filesystem::path sourcePath;

        /** @brief Default physics profile, model ID, integrator, and parameters. */
        PhysicsProfileConfig physicsProfile{};

        /** @brief Ordered list of bodies/devices/entities participating in this scene. */
        std::vector<BodyEntity> bodies;

        /** @brief Validates structural and physical invariants of the scene. */
        [[nodiscard]] bool Validate(std::string& outError) const;

        /** @brief Serializes this scene into canonical `.umgebung` JSON. */
        [[nodiscard]] nlohmann::json ToJson() const;

        /** @brief Deserializes and validates a scene from a `nlohmann::json` object. */
        [[nodiscard]] static bool FromJson(
            const nlohmann::json& jsonObj,
            UniverseScene& outScene,
            std::string& outError
        );

        /** @brief Reads, parses, and validates a `.umgebung` or `.scenario.umgebung` file from disk. */
        [[nodiscard]] static bool LoadFromFile(
            const std::filesystem::path& filePath,
            UniverseScene& outScene,
            std::string& outError
        );

        /** @brief Validates and writes this scene as formatted JSON to disk. */
        [[nodiscard]] bool SaveToFile(
            const std::filesystem::path& filePath,
            std::string& outError
        ) const;
    };

    /** @brief Semantic alias emphasizing shared core schema for standalone Scenarios. */
    using SimulationScene = UniverseScene;

} // namespace umgebung::scene
