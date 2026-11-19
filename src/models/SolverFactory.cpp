/**
 * @file SolverFactory.cpp
 * @brief Implementation of `umgebung::models::CreateSolverForScene`.
 */

#include "umgebung/models/SolverFactory.hpp"
#include "umgebung/models/mainstream/NewtonianNBodySolver.hpp"
#include "umgebung/models/unified/UnifiedFieldSolver.hpp"

namespace umgebung::models {

    std::unique_ptr<IPhysicsSolver> CreateSolverForScene(const scene::UniverseScene& scene) {
        switch (scene.physicsProfile.type) {
            case scene::PhysicsProfileType::Mainstream:
                return std::make_unique<mainstream::NewtonianNBodySolver>();
            case scene::PhysicsProfileType::Unified:
            case scene::PhysicsProfileType::Custom:
                return std::make_unique<unified::UnifiedFieldSolver>();
        }
        return std::make_unique<mainstream::NewtonianNBodySolver>();
    }

} // namespace umgebung::models
