/**
 * @file ScalarFieldSystem.hpp
 * @brief Contains the ScalarFieldSystem class.
 */
#pragma once

#include <entt/entt.hpp>

namespace Umgebung::renderer { class Camera; class DebugRenderer; }

namespace Umgebung::ecs::systems {

    /**
     * @brief A system that calculates the "Observer Effect" on the Scalar Field (Phryll).
     */
    class ScalarFieldSystem {
    public:
        /**
         * @brief Updates the system.
         * 
         * @param registry The EnTT registry.
         * @param camera The observer camera.
         * @param dt The delta time.
         */
        void onUpdate(entt::registry& registry, const renderer::Camera& camera, float dt);

        /**
         * @brief Visualizes the scalar field ripples in the debug renderer.
         */
        void visualize(entt::registry& registry, renderer::DebugRenderer* debugRenderer);
    };

}