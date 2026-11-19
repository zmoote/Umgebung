/**
 * @file SolverFactory.cpp
 * @brief Implementation of `umgebung::models::CreateSolverForScene` delegating
 *        cleanly to domain-aware `ModelRegistry::CreateSolver()`.
 */

#include "umgebung/models/SolverFactory.hpp"
#include "umgebung/models/ModelRegistry.hpp"

namespace umgebung::models {

    std::unique_ptr<IPhysicsSolver> CreateSolverForScene(
        const scene::UniverseScene& scene,
        std::size_t streamIndex
    ) {
        return ModelRegistry::CreateSolver(
            scene.physicsProfile.defaultModel,
            scene.domain,
            streamIndex
        );
    }

} // namespace umgebung::models
