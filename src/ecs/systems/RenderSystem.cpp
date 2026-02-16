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

    void RenderSystem::onUpdate(scene::Scene& scene, const renderer::Camera& camera, entt::entity selectedEntity, components::ScaleType observerScale) {
        auto& defaultShader = renderer_->getShader();
        auto& pointShader = renderer_->getPointShader();

        // Pre-set view/projection for both shaders
        defaultShader.bind();
        defaultShader.setMat4("view", camera.getViewMatrix());
        defaultShader.setMat4("projection", camera.getProjectionMatrix());
        defaultShader.setBool("uSelected", false);

        pointShader.bind();
        pointShader.setMat4("view", camera.getViewMatrix());
        pointShader.setMat4("projection", camera.getProjectionMatrix());
        pointShader.setBool("uSelected", false);

        auto& registry = scene.getRegistry();
        auto view = registry.view<components::Transform, components::Renderable>();

        // Simple state tracking to minimize shader switches
        bool usingPointShader = false;
        // Start with default
        defaultShader.bind(); 

        for (auto [entity, transform, renderable] : view.each()) {
            bool usePoints = false;
            auto* scaleComp = registry.try_get<components::ScaleComponent>(entity);
            
            if (scaleComp) {
                // Large scale objects are always points
                if (scaleComp->type >= components::ScaleType::Galactic) {
                    usePoints = true;
                }
                
                // If entity is smaller than current observer scale, and it's selected, force point rendering
                if (entity == selectedEntity && scaleComp->type < observerScale) {
                    usePoints = true;
                }
            }

            bool isSelected = (entity == selectedEntity);

            if (usePoints) {
                if (!usingPointShader) {
                    pointShader.bind();
                    usingPointShader = true;
                }
                
                pointShader.setMat4("model", transform.getModelMatrix());
                pointShader.setVec4("uColor", isSelected ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) : renderable.color);
                pointShader.setBool("uSelected", isSelected);
                
                // Use the shared point mesh for rendering
                renderer_->getPointMesh()->draw();
            } else {
                if (usingPointShader) {
                    defaultShader.bind();
                    usingPointShader = false;
                }

                if (renderable.mesh) {
                    defaultShader.setMat4("model", transform.getModelMatrix());
                    defaultShader.setVec4("uColor", isSelected ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) : renderable.color);
                    defaultShader.setBool("uSelected", isSelected);
                    renderable.mesh->draw();
                }
            }
        }
    }

} // namespace Umgebung::ecs::systems