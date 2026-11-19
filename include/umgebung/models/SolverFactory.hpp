/**
 * @file SolverFactory.hpp
 * @brief Convenience factory delegating `.umgebung` scene solver dispatch to
 *        the three-tier `umgebung::models::ModelRegistry`.
 */

#pragma once

#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <memory>

namespace umgebung::models {

    /**
     * @brief Instantiates the concrete `IPhysicsSolver` matching
     *        `scene.physicsProfile.defaultModel` via `ModelRegistry`.
     *
     * @param scene       Scene supplying `physicsProfile.defaultModel`.
     * @param streamIndex Optional CUDA stream pool index (`0` = Primary, `1` = Overlay).
     * @return Owning `std::unique_ptr<IPhysicsSolver>` (caller must invoke `Initialize()`).
     */
    [[nodiscard]] std::unique_ptr<IPhysicsSolver> CreateSolverForScene(
        const scene::UniverseScene& scene,
        std::size_t streamIndex = 0U
    );

} // namespace umgebung::models
