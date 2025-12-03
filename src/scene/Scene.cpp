/**
 * @file Scene.cpp
 * @brief Implements the Scene class.
 */
#include "umgebung/scene/Scene.hpp"

#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/util/LogMacros.hpp"

namespace Umgebung::scene {

    Scene::Scene() {

    }

    Scene::~Scene() {

    }

    entt::entity Scene::createEntity() {
        auto entity = registry_.create();
        registry_.emplace<ecs::components::Transform>(entity);
        registry_.emplace<ecs::components::Name>(entity, "Entity");
        UMGEBUNG_LOG_INFO("Created entity: {}", static_cast<uint32_t>(entity));
        return entity;
    }

    void Scene::destroyEntity(entt::entity entity) {
        UMGEBUNG_LOG_INFO("Destroyed entity: {}", static_cast<uint32_t>(entity));
        registry_.destroy(entity);
        if(m_SelectedEntity == entity) {
            m_SelectedEntity = entt::null;
        }
    }

    void Scene::onUpdate(float ts) {

    }

}