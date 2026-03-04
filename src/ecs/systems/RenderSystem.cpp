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
#include "umgebung/ecs/components/TimeComponent.hpp"
#include "umgebung/util/LogMacros.hpp"

namespace Umgebung::ecs::systems {

    RenderSystem::RenderSystem(renderer::Renderer* renderer)
        : renderer_(renderer) {
    }

    void RenderSystem::onUpdate(scene::Scene& scene, const renderer::Camera& camera, float time, entt::entity selectedEntity, components::ScaleType observerScale) {
        auto& defaultShader = renderer_->getShader();
        auto& pointShader = renderer_->getPointShader();

        // Pre-set view/projection for both shaders
        defaultShader.bind();
        defaultShader.setMat4("view", camera.getViewMatrix());
        defaultShader.setMat4("projection", camera.getProjectionMatrix());
        defaultShader.setVec3("uViewPos", camera.getPosition());
        defaultShader.setFloat("uTime", time);
        defaultShader.setBool("uSelected", false);
        defaultShader.setBool("uSourceView", sourceViewEnabled_);

        pointShader.bind();
        pointShader.setMat4("view", camera.getViewMatrix());
        pointShader.setMat4("projection", camera.getProjectionMatrix());
        pointShader.setFloat("uTime", time);
        pointShader.setBool("uSelected", false);
        pointShader.setBool("uSourceView", sourceViewEnabled_);

        auto& registry = scene.getRegistry();
        auto view = registry.view<components::Transform, components::Renderable>();

        // Simple state tracking to minimize shader switches
        bool usingPointShader = false;
        // Start with default
        defaultShader.bind(); 

        int renderedCount = 0;
        int pointCount = 0;

        for (auto [entity, transform, renderable] : view.each()) {
            bool usePoints = false;
            auto* scaleComp = registry.try_get<components::ScaleComponent>(entity);
            auto* timeComp = registry.try_get<components::TimeComponent>(entity);
            float density = timeComp ? timeComp->density : 3.0f;
            
            if (scaleComp) {
                // Only transition to point sprites at the Multiversal scale (cluster of universes)
                if (scaleComp->type >= components::ScaleType::Multiversal) {
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
                pointShader.setFloat("uDensity", density);
                
                renderer_->getPointMesh()->draw();
                pointCount++;
            } else {
                if (usingPointShader) {
                    defaultShader.bind();
                    usingPointShader = false;
                }

                if (renderable.mesh) {
                    defaultShader.setMat4("model", transform.getModelMatrix());
                    defaultShader.setVec4("uColor", isSelected ? glm::vec4(1.0f, 1.0f, 0.0f, 1.0f) : renderable.color);
                    defaultShader.setBool("uSelected", isSelected);
                    defaultShader.setFloat("uDensity", density);
                    renderable.mesh->draw();
                    renderedCount++;
                }
            }
        }

        static int lastRenderedCount = -1;
        static int lastPointCount = -1;
        static bool lastSourceView = false;

        if (renderedCount != lastRenderedCount || pointCount != lastPointCount || sourceViewEnabled_ != lastSourceView) {
            UMGEBUNG_LOG_TRACE("RenderSystem State Change: Meshes: {}, Points: {}, SourceView: {}", 
                renderedCount, pointCount, sourceViewEnabled_);
            lastRenderedCount = renderedCount;
            lastPointCount = pointCount;
            lastSourceView = sourceViewEnabled_;
        }
    }

} // namespace Umgebung::ecs::systems