/**
 * @file SolverFactory.hpp
 * @brief Factory dispatching `.umgebung` `UniverseScene` physics profiles to
 *        the appropriate concrete `IPhysicsSolver` implementation.
 *
 * @details
 * Centralizes Pillar 1 (`models/mainstream/`) vs. Pillar 2 (`models/unified/`)
 * solver selection so `core::Application` and future batch runners stay
 * completely decoupled from concrete solver types.
 */

#pragma once

#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <memory>

namespace umgebung::models {

    /**
     * @brief Instantiates the concrete `IPhysicsSolver` matching
     *        `scene.physicsProfile.type`.
     *
     * @details
     *   - `PhysicsProfileType::Mainstream` -> `mainstream::NewtonianNBodySolver`
     *   - `PhysicsProfileType::Unified`    -> `unified::UnifiedFieldSolver`
     *   - `PhysicsProfileType::Custom`     -> `unified::UnifiedFieldSolver`
     *
     * @param scene Universe scene whose `physicsProfile.type` selects the solver.
     * @return Owning `std::unique_ptr<IPhysicsSolver>` (caller must invoke `Initialize()`).
     */
    [[nodiscard]] std::unique_ptr<IPhysicsSolver> CreateSolverForScene(
        const scene::UniverseScene& scene
    );

} // namespace umgebung::models
