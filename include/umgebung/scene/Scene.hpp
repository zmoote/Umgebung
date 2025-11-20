/**
 * @file Scene.hpp
 * @brief Contains the Scene class.
 */
#pragma once

#include <entt/entt.hpp>

namespace Umgebung::scene {

    /**
     * @brief A class that manages the ECS registry and entities.
     */
    class Scene {
    public:
        /**
         * @brief Construct a new Scene object.
         */
        Scene();

        /**
         * @brief Destroy the Scene object.
         */
        ~Scene();

        /**
         * @brief Creates a new entity.
         * 
         * @return entt::entity 
         */
        entt::entity createEntity();

        /**
         * @brief Destroys an entity.
         * 
         * @param entity The entity to destroy.
         */
        void destroyEntity(entt::entity entity);

        /**
         * @brief Called every frame to update the scene.
         * 
         * @param ts The timestep.
         */
        void onUpdate(float ts);

        /**
         * @brief Get the Registry object.
         * 
         * @return entt::registry& 
         */
        entt::registry& getRegistry() { return registry_; }

        /**
         * @brief Set the Selected Entity object.
         * 
         * @param entity The entity to select.
         */
        void setSelectedEntity(entt::entity entity) { m_SelectedEntity = entity; }

        /**
         * @brief Get the Selected Entity object.
         * 
         * @return entt::entity 
         */
        entt::entity getSelectedEntity() const { return m_SelectedEntity; }

    private:
        entt::registry registry_; ///< The ECS registry.

        entt::entity m_SelectedEntity{ entt::null }; ///< The currently selected entity.
    };

}