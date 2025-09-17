#pragma once

#include <entt/entt.hpp>

namespace Umgebung::scene {

    /**
     * @class Scene
     * @brief Manages all entities, components, and systems in the world.
     *
     * The Scene is the container for a world's data. It uses an entt::registry
     * to manage the creation, destruction, and querying of entities and their
     * associated components.
     */
    class Scene {
    public:
        Scene();
        ~Scene();

        /**
         * @brief Creates a new entity in the scene.
         * @return The newly created entity handle.
         */
        entt::entity createEntity();

        /**
         * @brief Updates the scene's state.
         * @param ts The delta time since the last frame (timestep).
         */
        void onUpdate(float ts);

        /**
         * @brief Gets a reference to the underlying ECS registry.
         * @return A reference to the entt::registry.
         */
        entt::registry& getRegistry() { return registry_; }

    private:
        entt::registry registry_;
    };

} // namespace Umgebung::scene