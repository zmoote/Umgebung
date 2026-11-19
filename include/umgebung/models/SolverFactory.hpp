/**
 * @file SolverFactory.hpp
 * @brief Convenience factory delegating `.umgebung` scene solver dispatch to
 *        the domain-filtered `umgebung::models::ModelRegistry`.
 */

#pragma once

#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <memory>

namespace umgebung::models {

    /**
     * @brief Instantiates the concrete `IPhysicsSolver` matching both
     *        `scene.physicsProfile.defaultModel` and `scene.domain`.
     */
    [[nodiscard]] std::unique_ptr<IPhysicsSolver> CreateSolverForScene(
        const scene::UniverseScene& scene,
        std::size_t streamIndex = 0U
    );

} // namespace umgebung::models
