/**
 * @file ModelRegistry.hpp
 * @brief Composable three-tier Model Registry enumerating and instantiating
 *        every physics solver across Tier 1 (`Mainstream`), Tier 2 (standalone
 *        `Thinker::*`), and Tier 3 (`Unified::Synthesis`).
 *
 * @details
 * Powers:
 *   - Headless CLI `--model <name>` and `--compare-model <name>`,
 *   - Dear ImGui live Primary Model A and Comparison Overlay Model B dropdowns,
 *   - Automated CTest matrices sweeping any Scenario across multiple models.
 */

#pragma once

#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/PhysicsProfile.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace umgebung::models {

    /**
     * @struct ModelDescriptor
     * @brief Metadata describing a single solver registered in `ModelRegistry`.
     */
    struct ModelDescriptor {
        /** @brief Canonical enum identifier. */
        scene::ModelId id{scene::ModelId::MainstreamNewtonian};

        /** @brief Exact CLI / `.umgebung` string (e.g., `"Mainstream::DarkMatterNFW"`). */
        const char* canonicalName{""};

        /** @brief Human-readable label shown in UI dropdowns. */
        const char* displayName{""};

        /** @brief Tier classification (`"Tier 1: Mainstream"`, `"Tier 2: Thinker"`, `"Tier 3: Unified"`). */
        const char* tierLabel{""};

        /** @brief Concise bibliographic / research source reference. */
        const char* primarySource{""};
    };

    /**
     * @class ModelRegistry
     * @brief Central static registry for discovering and instantiating all 8
     *        Umgebung simulation models.
     */
    class ModelRegistry final {
    public:
        /** @brief Returns the ordered list of all 8 registered model descriptors. */
        [[nodiscard]] static const std::vector<ModelDescriptor>& GetAllModels() noexcept;

        /** @brief Finds a descriptor by `ModelId` (returns `nullptr` if unknown). */
        [[nodiscard]] static const ModelDescriptor* FindById(scene::ModelId id) noexcept;

        /** @brief Finds a descriptor by canonical name or short alias (returns `nullptr` if unknown). */
        [[nodiscard]] static const ModelDescriptor* FindByName(const std::string& name) noexcept;

        /**
         * @brief Instantiates a fresh concrete `IPhysicsSolver` for `id` bound
         *        to the requested CUDA `streamIndex` (`0` = Primary Model A,
         *        `1` = Simultaneous Overlay Model B).
         *
         * @param id          Canonical model identifier.
         * @param streamIndex CUDA stream pool index passed to `SetStreamIndex()`.
         * @return Owning `std::unique_ptr<IPhysicsSolver>` ready for `Initialize()`.
         */
        [[nodiscard]] static std::unique_ptr<IPhysicsSolver> CreateSolver(
            scene::ModelId id,
            std::size_t streamIndex = 0U
        );
    };

} // namespace umgebung::models
