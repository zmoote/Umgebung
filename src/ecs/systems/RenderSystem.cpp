/**
 * @file RenderSystem.cpp
 * @brief Implements the RenderSystem class.
 */
#include "umgebung/ecs/systems/RenderSystem.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"

#include <glad/glad.h>
#include <entt/entt.hpp>
#include <glm/glm.hpp>

namespace Umgebung::ecs::systems {

    RenderSystem::RenderSystem(renderer::Renderer* renderer)
        : renderer_(renderer) {
    }

    void RenderSystem::onUpdate(scene::Scene& scene) {
        auto& shader = renderer_->getShader();
        shader.bind();

        // --- FIX: Pass both name and matrix ---
        shader.setMat4("view", renderer_->getViewMatrix());
        shader.setMat4("projection", renderer_->getProjectionMatrix());
        // --- END FIX ---

        auto& registry = scene.getRegistry();

        // Use the component names you refactored to: Transform and Renderable
        auto view = registry.view<components::Transform, components::Renderable>();

        for (auto [entity, transform, renderable] : view.each()) {
            if (renderable.mesh) {
                // Set model matrix
                shader.setMat4("model", transform.getModelMatrix());

                // Set color (assuming you add a "u_Color" uniform to your shader)
                shader.setVec4("uColor", renderable.color); 

                // Bind and draw the mesh
                renderable.mesh->draw();
            }
        }
    }

} // namespace Umgebung::ecs::systems