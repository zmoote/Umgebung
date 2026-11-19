/**
 * @file ModelRegistry.hpp
 * @brief Domain-filtered three-tier Model Registry enumerating and instantiating
 *        every physics solver across Tier 1 (`Mainstream`), Tier 2 (standalone
 *        `Thinker::*`), and Tier 3 (`Unified::Synthesis`), strictly enforcing
 *        `SimulationDomain` compatibility so non-orbital problems are never
 *        paired with orbital N-body solvers (or vice versa).
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
     * @enum ModelProvenance
     * @brief Explicit epistemological classification separating verified textbook
     *        physics from verbatim Thinker equations and phenomenological ansatzes.
     *
     * @details
     * Ensures complete scientific transparency in CLI logs, JSON reports, and
     * Dear ImGui panels so researchers and reviewers immediately distinguish
     * exact published equations from operationalized hypotheses.
     */
    enum class ModelProvenance : std::uint8_t {
        /** @brief Standard, academically verified textbook physics baseline. */
        TextbookBaseline = 0,
        /** @brief Closed-form equations taken verbatim from the Thinker's primary papers. */
        VerbatimEquation = 1,
        /** @brief Verbatim published equations/hardware specs paired with a parameterized coupling ansatz. */
        HybridVerbatimAndAnsatz = 2,
        /** @brief Qualitative cosmological framework operationalized into well-posed mathematical ansatzes. */
        OperationalizedSpec = 3
    };

    /** @brief Converts `ModelProvenance` into a concise human-readable badge string. */
    [[nodiscard]] inline const char* ModelProvenanceToString(ModelProvenance provenance) noexcept {
        switch (provenance) {
            case ModelProvenance::TextbookBaseline:        return "Textbook Baseline";
            case ModelProvenance::VerbatimEquation:        return "Verbatim Equations";
            case ModelProvenance::HybridVerbatimAndAnsatz: return "Verbatim + Ansatz";
            case ModelProvenance::OperationalizedSpec:     return "Operationalized Spec / Ansatz";
        }
        return "Unknown";
    }

    /**
     * @struct ModelDescriptor
     * @brief Metadata, `SimulationDomainMask`, and epistemological provenance
     *        describing a single solver registered in `ModelRegistry`.
     */
    struct ModelDescriptor {
        scene::ModelId id{scene::ModelId::MainstreamNewtonian};
        const char* canonicalName{""};
        const char* displayName{""};
        const char* tierLabel{""};
        const char* primarySource{""};

        /** @brief Bitmask of `SimulationDomain`s physically supported by this model. */
        scene::SimulationDomainMask supportedDomains{scene::kDomainNBodyDynamics};

        /** @brief Epistemological provenance badge (Textbook vs Verbatim vs Ansatz). */
        ModelProvenance provenance{ModelProvenance::TextbookBaseline};

        /** @brief Explicit breakdown of what is verbatim from the source vs. parameterized ansatz. */
        const char* provenanceDetails{""};
    };

    /**
     * @class ModelRegistry
     * @brief Central static registry for discovering, filtering by domain, and
     *        instantiating all 10 Umgebung simulation models.
     */
    class ModelRegistry final {
    public:
        /** @brief Returns all registered model descriptors across every domain. */
        [[nodiscard]] static const std::vector<ModelDescriptor>& GetAllModels() noexcept;

        /** @brief Returns strictly the models compatible with `domain`. */
        [[nodiscard]] static std::vector<ModelDescriptor> GetModelsForDomain(
            scene::SimulationDomain domain
        );

        /** @brief Returns `true` iff `id` physically supports `domain`. */
        [[nodiscard]] static bool IsModelCompatibleWithDomain(
            scene::ModelId id,
            scene::SimulationDomain domain
        ) noexcept;

        /** @brief Returns a recommended default `ModelId` for `domain`. */
        [[nodiscard]] static scene::ModelId GetDefaultModelForDomain(
            scene::SimulationDomain domain
        ) noexcept;

        /** @brief Finds a descriptor by `ModelId` (`nullptr` if unknown). */
        [[nodiscard]] static const ModelDescriptor* FindById(scene::ModelId id) noexcept;

        /** @brief Finds a descriptor by canonical name or alias (`nullptr` if unknown). */
        [[nodiscard]] static const ModelDescriptor* FindByName(const std::string& name) noexcept;

        /**
         * @brief Instantiates a fresh concrete `IPhysicsSolver` implementing `id`
         *        within `domain`, bound to `streamIndex` (`0` = Primary, `1` = Overlay).
         *
         * @return Valid `std::unique_ptr<IPhysicsSolver>`, or `nullptr` if `id`
         *         does not support `domain`.
         */
        [[nodiscard]] static std::unique_ptr<IPhysicsSolver> CreateSolver(
            scene::ModelId id,
            scene::SimulationDomain domain,
            std::size_t streamIndex = 0U
        );

        /**
         * @brief Convenience overload defaulting to `SimulationDomain::NBodyDynamics`
         *        (or the model's primary domain if it does not support `NBodyDynamics`).
         */
        [[nodiscard]] static std::unique_ptr<IPhysicsSolver> CreateSolver(
            scene::ModelId id,
            std::size_t streamIndex = 0U
        );
    };

} // namespace umgebung::models
