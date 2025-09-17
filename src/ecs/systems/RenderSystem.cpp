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
        // 1. Get the ECS registry from the scene.
        auto& registry = scene.getRegistry();

        // 2. Get the main shader from the renderer.
        // NOTE: In the future, this would be determined by the material.
        auto& shader = renderer_->getShader();
        shader.bind();

        // 3. Set camera uniforms (View and Projection matrices).
        // These are the same for every object in the scene.
        shader.setMat4("view", renderer_->getViewMatrix());
        shader.setMat4("projection", renderer_->getProjectionMatrix());

        // 4. Create a view to get all entities that have BOTH a Transform and Renderable component.
        auto view = registry.view<const ecs::components::TransformComponent,
            const ecs::components::RenderableComponent>();

        // 5. Loop over each entity in the view.
        for (auto entity : view) {
            // Get the components from the entity.
            // The 'const' ensures we don't accidentally modify the data during rendering.
            const auto& transform = view.get<ecs::components::TransformComponent>(entity);
            const auto& renderable = view.get<ecs::components::RenderableComponent>(entity);

            // Skip if the entity doesn't have a mesh assigned.
            if (!renderable.mesh) {
                continue;
            }

            // 6. Set object-specific uniforms.
            shader.setMat4("model", transform.getModelMatrix());
            shader.setVec4("uColor", renderable.color);

            // 7. Tell the mesh to draw itself.
            renderable.mesh->draw();
        }

        shader.unbind();
    }

} // namespace Umgebung::ecs::systems