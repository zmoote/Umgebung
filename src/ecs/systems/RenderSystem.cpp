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
#include <glm/gtx/norm.hpp>
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

    void RenderSystem::onUpdate(scene::Scene& scene, const renderer::Camera& camera, float time, entt::entity selectedEntity, components::ScaleType observerScale, bool forceRebuild) {
        auto& defaultShader = renderer_->getShader();
        auto& pointShader = renderer_->getPointShader();

        auto& registry = scene.getRegistry();
        size_t currentRegistrySize = registry.storage<entt::entity>().size();

        // 1. Determine if we need to rebuild batches
        // We rebuild if the structure changed, selection changed, or scale changed
        bool needsRebuild = forceRebuild || scene.isDirty() ||
                            (currentRegistrySize != lastRegistrySize_) || 
                            (selectedEntity != lastSelectedEntity_) || 
                            (observerScale != lastObserverScale_) ||
                            (sourceViewEnabled_ != lastSourceViewEnabled_);

        // For now, we also rebuild if any Phryll components exist because they update every frame
        if (!needsRebuild) {
            auto phryllView = registry.view<components::PhryllComponent>();
            if (!phryllView.empty()) needsRebuild = true;
        }

        // 2. Setup Global Uniforms
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

        // 3. Prepare Batches (only if needed)
        if (needsRebuild) {
            scene.setDirty(false); // Reset dirty flag
            for (auto& [mesh, instances] : meshBatches_) {
                instances.clear();
            }
            pointBatch_.clear();

            auto view = registry.view<components::Transform, components::Renderable>();
            
            glm::vec3 cameraPos = camera.getPosition();
            glm::vec3 cameraForward = camera.getForward();

            for (auto [entity, transform, renderable] : view.each()) {
                bool usePoints = false;
                auto* scaleComp = registry.try_get<components::ScaleComponent>(entity);
                
                // Simple Frustum/Distance Culling
                // If the entity is at a scale much smaller than the observer and far away, skip it
                if (scaleComp) {
                    int scaleDiff = static_cast<int>(scaleComp->type) - static_cast<int>(observerScale);
                    
                    // LOD: Transition to points if too far or scale is too large
                    if (scaleComp->type >= components::ScaleType::Multiversal) {
                        usePoints = true;
                    } else if (scaleDiff > 1) {
                        usePoints = true;
                    } else if (scaleDiff < -2) {
                        // Too small to see, skip
                        continue;
                    }
                    
                    // Simple distance culling
                    float distSq = glm::distance2(transform.position, cameraPos);
                    float maxDist = camera.getFarPlane() * 0.9f;
                    if (distSq > maxDist * maxDist) continue;
                }

                auto* timeComp = registry.try_get<components::TimeComponent>(entity);
                auto* phryllComp = registry.try_get<components::PhryllComponent>(entity);

                float density = timeComp ? timeComp->density : 3.0f;
                float phryllInfluence = phryllComp ? phryllComp->observerInfluence : 0.0f;
                float isManifesting = phryllComp ? (phryllComp->isManifesting ? 1.0f : 0.0f) : 1.0f;

                renderer::InstanceData data;
                data.modelMatrix = transform.getModelMatrix();
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
            lastRegistrySize_ = currentRegistrySize;
        }

        // 4. Draw Mesh Batches
        defaultShader.bind();
        for (auto& [mesh, instances] : meshBatches_) {
            if (instances.empty()) continue;

            if (instances.size() > 1) {
                defaultShader.setBool("uIsInstanced", true);
                mesh->drawInstanced(instances, needsRebuild);
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

        // 5. Draw Point Batch
        if (!pointBatch_.empty()) {
            pointShader.bind();
            if (pointBatch_.size() > 1) {
                pointShader.setBool("uIsInstanced", true);
                renderer_->getPointMesh()->drawInstanced(pointBatch_, needsRebuild);
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

        static size_t lastReportedCount = 0;
        if (currentRegistrySize != lastReportedCount) {
            UMGEBUNG_LOG_TRACE("RenderSystem: Batching {} entities (Active). Rebuild: {}", 
                currentRegistrySize, needsRebuild);
            lastReportedCount = currentRegistrySize;
        }
    }

} // namespace Umgebung::ecs::systems