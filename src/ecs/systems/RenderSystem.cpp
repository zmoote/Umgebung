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

#include "umgebung/ecs/components/ScaleComponent.hpp"

namespace Umgebung::ecs::systems {

    RenderSystem::RenderSystem(renderer::Renderer* renderer)
        : renderer_(renderer) {
    }

    void RenderSystem::onUpdate(scene::Scene& scene, const renderer::Camera& camera) {
        auto& defaultShader = renderer_->getShader();
        auto& pointShader = renderer_->getPointShader();

        // Pre-set view/projection for both shaders
        defaultShader.bind();
        defaultShader.setMat4("view", camera.getViewMatrix());
        defaultShader.setMat4("projection", camera.getProjectionMatrix());

        pointShader.bind();
        pointShader.setMat4("view", camera.getViewMatrix());
        pointShader.setMat4("projection", camera.getProjectionMatrix());

        auto& registry = scene.getRegistry();
        auto view = registry.view<components::Transform, components::Renderable>();

        // Simple state tracking to minimize shader switches
        bool usingPointShader = false;
        // Start with default (though we bound point last, so let's ensure we bind the right one first thing)
        defaultShader.bind(); 

        for (auto [entity, transform, renderable] : view.each()) {
            bool usePoints = false;
            auto* scaleComp = registry.try_get<components::ScaleComponent>(entity);
            
            if (scaleComp && scaleComp->type >= components::ScaleType::Galactic) {
                usePoints = true;
            }

            if (usePoints) {
                if (!usingPointShader) {
                    pointShader.bind();
                    usingPointShader = true;
                }
                
                pointShader.setMat4("model", transform.getModelMatrix());
                pointShader.setVec4("uColor", renderable.color);
                
                // Use the shared point mesh for rendering
                renderer_->getPointMesh()->draw();
            } else {
                if (usingPointShader) {
                    defaultShader.bind();
                    usingPointShader = false;
                }

                if (renderable.mesh) {
                    defaultShader.setMat4("model", transform.getModelMatrix());
                    defaultShader.setVec4("uColor", renderable.color);
                    renderable.mesh->draw();
                }
            }
        }
    }

} // namespace Umgebung::ecs::systems