#include "umgebung/ecs/systems/RenderSystem.hpp"

#include "umgebung/scene/Scene.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"

namespace Umgebung::ecs::systems {

    RenderSystem::RenderSystem(renderer::Renderer* renderer)
        : renderer_(renderer) {
    }

    void RenderSystem::onUpdate(scene::Scene& scene) {
        auto& registry = scene.getRegistry();

        auto& shader = renderer_->getShader();
        shader.bind();

        shader.setMat4("view", renderer_->getViewMatrix());
        shader.setMat4("projection", renderer_->getProjectionMatrix());

        auto view = registry.view<const ecs::components::TransformComponent,
            const ecs::components::RenderableComponent>();

        for (auto entity : view) {
            const auto& transform = view.get<ecs::components::TransformComponent>(entity);
            const auto& renderable = view.get<ecs::components::RenderableComponent>(entity);

            if (!renderable.mesh) {
                continue;
            }

            shader.setMat4("model", transform.getModelMatrix());
            shader.setVec4("uColor", renderable.color);

            renderable.mesh->draw();
        }

        shader.unbind();
    }

}