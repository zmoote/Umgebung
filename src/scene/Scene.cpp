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

    void Scene::onUpdate(float ts) {

    }

}