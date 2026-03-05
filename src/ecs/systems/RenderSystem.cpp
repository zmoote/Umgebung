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
#include <unordered_map>
#include <vector>

#include "umgebung/ecs/components/ScaleComponent.hpp"
#include "umgebung/ecs/components/TimeComponent.hpp"
#include "umgebung/ecs/components/PhryllComponent.hpp"
#include "umgebung/util/LogMacros.hpp"

namespace Umgebung::ecs::systems {

    RenderSystem::RenderSystem(renderer::Renderer* renderer)
        : renderer_(renderer) {
    }

    void RenderSystem::onUpdate(scene::Scene& scene, const renderer::Camera& camera, float time, entt::entity selectedEntity, components::ScaleType observerScale) {
        auto& defaultShader = renderer_->getShader();
        auto& pointShader = renderer_->getPointShader();

        // 1. Setup Global Uniforms
        defaultShader.bind();
        defaultShader.setMat4("view", camera.getViewMatrix());
        defaultShader.setMat4("projection", camera.getProjectionMatrix());
        defaultShader.setVec3("uViewPos", camera.getPosition());
        defaultShader.setFloat("uTime", time);
        defaultShader.setBool("uSelected", false);
        defaultShader.setBool("uSourceView", sourceViewEnabled_);
        defaultShader.setBool("uIsInstanced", false);

        pointShader.bind();
        pointShader.setMat4("view", camera.getViewMatrix());
        pointShader.setMat4("projection", camera.getProjectionMatrix());
        pointShader.setFloat("uTime", time);
        pointShader.setBool("uSelected", false);
        pointShader.setBool("uSourceView", sourceViewEnabled_);
        pointShader.setBool("uIsInstanced", false);

        auto& registry = scene.getRegistry();

        // 2. Prepare Batches
        for (auto& [mesh, instances] : meshBatches_) {
            instances.clear();
        }
        pointBatch_.clear();

        auto view = registry.view<components::Transform, components::Renderable>();
        int totalEntities = 0;

        for (auto [entity, transform, renderable] : view.each()) {
            totalEntities++;
            bool usePoints = false;
            auto* scaleComp = registry.try_get<components::ScaleComponent>(entity);
            auto* timeComp = registry.try_get<components::TimeComponent>(entity);
            auto* phryllComp = registry.try_get<components::PhryllComponent>(entity);

            float density = timeComp ? timeComp->density : 3.0f;
            float phryllInfluence = phryllComp ? phryllComp->observerInfluence : 0.0f;
            float isManifesting = phryllComp ? (phryllComp->isManifesting ? 1.0f : 0.0f) : 1.0f;
            
            if (scaleComp) {
                if (scaleComp->type >= components::ScaleType::Multiversal) {
                    usePoints = true;
                }
                
                if (entity == selectedEntity && scaleComp->type < observerScale) {
                    usePoints = true;
                }
            }

            renderer::InstanceData data;
            data.modelMatrix = transform.getModelMatrix(); // Uses cache!
            data.color = renderable.color;
            data.density = density;
            data.phryllInfluence = phryllInfluence;
            data.selected = (entity == selectedEntity) ? 1.0f : 0.0f;
            data.isManifesting = isManifesting;

            if (usePoints) {
                pointBatch_.push_back(data);
            } else if (renderable.mesh) {
                meshBatches_[renderable.mesh.get()].push_back(data);
            }
        }

        lastSelectedEntity_ = selectedEntity;
        lastObserverScale_ = observerScale;
        lastSourceViewEnabled_ = sourceViewEnabled_;
        lastRegistrySize_ = registry.alive();

        // 3. Draw Mesh Batches
        defaultShader.bind();
        for (auto& [mesh, instances] : meshBatches_) {
            if (instances.empty()) continue;

            if (instances.size() > 1) {
                defaultShader.setBool("uIsInstanced", true);
                mesh->drawInstanced(instances);
            } else {
                defaultShader.setBool("uIsInstanced", false);
                const auto& data = instances[0];
                defaultShader.setMat4("model", data.modelMatrix);
                defaultShader.setVec4("uColor", data.color);
                defaultShader.setBool("uSelected", data.selected > 0.5f);
                defaultShader.setFloat("uDensity", data.density);
                defaultShader.setFloat("uPhryllInfluence", data.phryllInfluence);
                defaultShader.setBool("uIsManifesting", data.isManifesting > 0.5f);
                mesh->draw();
            }
        }

        // 4. Draw Point Batch
        if (!pointBatch_.empty()) {
            pointShader.bind();
            if (pointBatch_.size() > 1) {
                pointShader.setBool("uIsInstanced", true);
                renderer_->getPointMesh()->drawInstanced(pointBatch_);
            } else {
                pointShader.setBool("uIsInstanced", false);
                const auto& data = pointBatch_[0];
                pointShader.setMat4("model", data.modelMatrix);
                pointShader.setVec4("uColor", data.color);
                pointShader.setBool("uSelected", data.selected > 0.5f);
                pointShader.setFloat("uDensity", data.density);
                pointShader.setFloat("uPhryllInfluence", data.phryllInfluence);
                pointShader.setBool("uIsManifesting", data.isManifesting > 0.5f);
                renderer_->getPointMesh()->draw();
            }
        }

        static int lastTotalEntities = -1;
        if (totalEntities != lastTotalEntities) {
            UMGEBUNG_LOG_TRACE("RenderSystem: Batching {} entities into {} mesh groups and 1 point group.", 
                totalEntities, meshBatches_.size());
            lastTotalEntities = totalEntities;
        }
    }

} // namespace Umgebung::ecs::systems