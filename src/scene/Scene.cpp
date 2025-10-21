#include "umgebung/scene/Scene.hpp"

#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"

namespace Umgebung::scene {

    Scene::Scene() {

    }

    Scene::~Scene() {

    }

    entt::entity Scene::createEntity() {
        auto entity = registry_.create();
        registry_.emplace<ecs::components::TransformComponent>(entity);
        return entity;
    }

    void Scene::destroyEntity(entt::entity entity) {
        registry_.destroy(entity);
        if(m_SelectedEntity == entity) {
            m_SelectedEntity = entt::null;
        }
    }

    void Scene::onUpdate(float ts) {

    }

}