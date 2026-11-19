/**
 * @file UniverseScene.hpp
 * @brief Domain model and `.umgebung` JSON v1 serializer for a single Universe
 *        within the Umgebung Multiverse.
 *
 * @details
 * Implements the per-universe `.umgebung` scene file specification:
 *   - Every universe in the multiverse resides in its own `.umgebung` JSON file.
 *   - Encodes schema versioning (`schema_version = 1`), universe metadata,
 *     the active `PhysicsProfileConfig` (`Mainstream`, `Unified`, or `Custom`),
 *     and a hierarchical list of physical/conscious `BodyEntity` objects
 *     (linked via `parentId` to represent Galaxy -> Star -> Planet -> Moon or
 *     subatomic nesting inside the hierarchy inspector).
 *   - Uses IEEE-754 double precision (`double`) throughout so initial
 *     conditions preserve full 15-17 significant decimal digits for analytical
 *     benchmarks and conservation-law verification.
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
     *
     * @details
     * Used for host-side initial conditions, invariant diagnostics (momentum,
     * angular momentum, center of mass), and `.umgebung` JSON serialization.
     */
    struct Vec3d {
        /** @brief X-axis Cartesian coordinate/component. */
        double x{0.0};
        /** @brief Y-axis Cartesian coordinate/component. */
        double y{0.0};
        /** @brief Z-axis Cartesian coordinate/component. */
        double z{0.0};

        /** @brief Vector addition `a + b`. */
        [[nodiscard]] constexpr Vec3d operator+(const Vec3d& rhs) const noexcept {
            return {x + rhs.x, y + rhs.y, z + rhs.z};
        }

        /** @brief Vector subtraction `a - b`. */
        [[nodiscard]] constexpr Vec3d operator-(const Vec3d& rhs) const noexcept {
            return {x - rhs.x, y - rhs.y, z - rhs.z};
        }

        /** @brief Scalar multiplication `v * s`. */
        [[nodiscard]] constexpr Vec3d operator*(double scalar) const noexcept {
            return {x * scalar, y * scalar, z * scalar};
        }

        /** @brief Euclidean inner (dot) product $\mathbf{a} \cdot \mathbf{b}$. */
        [[nodiscard]] static constexpr double Dot(const Vec3d& a, const Vec3d& b) noexcept {
            return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
        }

        /** @brief 3D cross product $\mathbf{a} \times \mathbf{b}$ (used for angular momentum $\mathbf{r} \times \mathbf{p}$). */
        [[nodiscard]] static constexpr Vec3d Cross(const Vec3d& a, const Vec3d& b) noexcept {
            return {
                (a.y * b.z) - (a.z * b.y),
                (a.z * b.x) - (a.x * b.z),
                (a.x * b.y) - (a.y * b.x)
            };
        }

        /** @brief Squared Euclidean norm $\|\mathbf{v}\|^2 = x^2 + y^2 + z^2$. */
        [[nodiscard]] constexpr double NormSquared() const noexcept {
            return Dot(*this, *this);
        }

        /** @brief Euclidean magnitude $\|\mathbf{v}\| = \sqrt{x^2 + y^2 + z^2}$. */
        [[nodiscard]] double Norm() const noexcept {
            return std::sqrt(NormSquared());
        }
    };

    /**
     * @struct BodyEntity
     * @brief Persistent descriptor for a single gravitating/physical/conscious
     *        entity inside a `UniverseScene`.
     *
     * @details
     * Stores both classical mechanical state (`mass`, `radius`, `position`,
     * `velocity`) used by `models/mainstream/` and optional metaphysical /
     * field properties (`vibrationalDensity`, `consciousnessCoherence`,
     * `phryllCharge`) preserved seamlessly for `models/unified/`.
     */
    struct BodyEntity {
        /** @brief Unique identifier within the enclosing universe (e.g., `"star_sol"`, `"body_1"`). */
        std::string id;

        /** @brief Human-readable display name shown in logs and the Multiverse Hierarchy panel. */
        std::string name;

        /** @brief Semantic classification (e.g., `"Star"`, `"Planet"`, `"Particle"`, `"Crystal"`). */
        std::string category{"Body"};

        /** @brief Optional `id` of a parent entity within the same universe (`""` = root-level entity). */
        std::string parentId;

        /** @brief Inertial/gravitational mass $m > 0$ in scene mass units. */
        double mass{1.0};

        /** @brief Characteristic physical radius $R \ge 0$ in scene length units. */
        double radius{0.05};

        /** @brief Initial Cartesian position $\mathbf{r}_0 = (x, y, z)$ in scene length units. */
        Vec3d position{};

        /** @brief Initial Cartesian velocity $\mathbf{v}_0 = (v_x, v_y, v_z)$ in scene length/time units. */
        Vec3d velocity{};

        // ---------------------------------------------------------------------
        // Optional Unified Research Model Properties (Ignored by Mainstream)
        // ---------------------------------------------------------------------

        /** @brief Entity vibrational density level (default `3.0`). */
        double vibrationalDensity{3.0};

        /** @brief Consciousness coherence scalar $[0, 1]$ for `models/unified/` coupling. */
        double consciousnessCoherence{0.0};

        /** @brief Stored/emitted Phryll life-force energy charge for `models/unified/`. */
        double phryllCharge{0.0};
    };

    /**
     * @struct UniverseScene
     * @brief Complete in-memory representation of a `.umgebung` universe scene file.
     */
    struct UniverseScene {
        /** @brief Current `.umgebung` JSON schema version supported by this engine. */
        static constexpr std::uint32_t kCurrentSchemaVersion = 1U;

        /** @brief Schema version parsed from JSON (must equal `kCurrentSchemaVersion`). */
        std::uint32_t schemaVersion{kCurrentSchemaVersion};

        /** @brief Stable machine-friendly identifier for this universe (e.g., `"kepler_two_body"`). */
        std::string universeId;

        /** @brief Human-readable universe display title. */
        std::string name;

        /** @brief Scientific or cosmological description of the scene and its expected behavior. */
        std::string description;

        /** @brief Filesystem path from which this scene was loaded (empty for programmatic scenes). */
        std::filesystem::path sourcePath;

        /** @brief Active physics profile, integrator, units, and force/field parameters. */
        PhysicsProfileConfig physicsProfile{};

        /** @brief Ordered list of bodies/entities participating in this universe. */
        std::vector<BodyEntity> bodies;

        /**
         * @brief Validates structural and physical invariants of the scene.
         *
         * @details
         * Verifies:
         *   - Supported `schemaVersion`,
         *   - Non-empty `universeId` and `name`,
         *   - Strictly finite, positive `physicsProfile.timestep` and non-negative softening,
         *   - Every entity has a unique non-empty `id`, finite positive `mass`,
         *     and finite `position`/`velocity` components.
         *
         * @param[out] outError Populated with a human-readable explanation if invalid.
         * @return `true` if all invariants hold; `false` otherwise.
         */
        [[nodiscard]] bool Validate(std::string& outError) const;

        /**
         * @brief Serializes this `UniverseScene` into canonical `.umgebung` JSON.
         *
         * @return `nlohmann::json` object adhering to schema version `1`.
         */
        [[nodiscard]] nlohmann::json ToJson() const;

        /**
         * @brief Deserializes and validates a `UniverseScene` from a JSON object.
         *
         * @param      jsonObj  Parsed `nlohmann::json` document.
         * @param[out] outScene Populated `UniverseScene` on success.
         * @param[out] outError Human-readable diagnostic message on failure.
         * @return `true` if parsing and validation succeeded; `false` otherwise.
         */
        [[nodiscard]] static bool FromJson(
            const nlohmann::json& jsonObj,
            UniverseScene& outScene,
            std::string& outError
        );

        /**
         * @brief Reads, parses, and validates a `.umgebung` JSON file from disk.
         *
         * @param      filePath Path to a `.umgebung` file.
         * @param[out] outScene Populated `UniverseScene` (including `sourcePath`) on success.
         * @param[out] outError Human-readable diagnostic message on failure.
         * @return `true` on success; `false` on I/O, JSON syntax, or validation error.
         */
        [[nodiscard]] static bool LoadFromFile(
            const std::filesystem::path& filePath,
            UniverseScene& outScene,
            std::string& outError
        );

        /**
         * @brief Validates and writes this `UniverseScene` as formatted JSON to disk.
         *
         * @param      filePath Destination `.umgebung` file path.
         * @param[out] outError Human-readable diagnostic message on failure.
         * @return `true` on success; `false` on validation or filesystem error.
         */
        [[nodiscard]] bool SaveToFile(
            const std::filesystem::path& filePath,
            std::string& outError
        ) const;
    };

} // namespace umgebung::scene
