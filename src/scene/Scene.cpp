#include "umgebung/scene/Scene.hpp"

#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"

namespace Umgebung::scene {

    Scene::Scene() {
        // In the future, you might initialize scene-wide resources here.
    }

    Scene::~Scene() {
        // Clean up any resources. The registry handles component destruction automatically.
    }

    entt::entity Scene::createEntity() {
        // Create a new entity. By default, we'll give it a TransformComponent.
        auto entity = registry_.create();
        registry_.emplace<ecs::components::TransformComponent>(entity);
        return entity;
    }

    void Scene::onUpdate(float ts) {
        // This is where you will update your systems.
        // For example:
        // physicsSystem.update(ts, registry_);
        // renderSystem.update(ts, registry_);
        // scriptSystem.update(ts, registry_);
    }

} // namespace Umgebung::scene