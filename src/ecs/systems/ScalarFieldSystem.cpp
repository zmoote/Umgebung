/**
 * @file ScalarFieldSystem.cpp
 * @brief Implements the ScalarFieldSystem class.
 */
#include "umgebung/ecs/systems/ScalarFieldSystem.hpp"
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/renderer/DebugRenderer.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/PhryllComponent.hpp"
#include "umgebung/ecs/components/TimeComponent.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <entt/entt.hpp>

namespace Umgebung::ecs::systems {

    void ScalarFieldSystem::onUpdate(entt::registry& registry, const renderer::Camera& camera, float dt) {
        auto view = registry.view<components::Transform, components::PhryllComponent>();

        glm::vec3 cameraPos = camera.getPosition();
        glm::vec3 cameraForward = camera.getForward();

        for (auto [entity, transform, phryll] : view.each()) {
            // 1. Calculate Observer Proximity Influence
            glm::vec3 toEntity = transform.position - cameraPos;
            float distSq = glm::dot(toEntity, toEntity);
            
            // Optimization: Skip calculation for extremely far objects at current scale
            // Threshold depends on scale, but for now let's use a fixed relative threshold
            float farThreshold = camera.getFarPlane() * 0.5f;
            if (distSq > farThreshold * farThreshold) {
                phryll.observerInfluence = 0.0f;
                continue;
            }

            float distance = sqrt(distSq);
            if (distance < 0.0001f) distance = 0.0001f;
            
            glm::vec3 dirToEntity = toEntity / distance;

            // 2. Calculate Observer Focus (The "Observer Effect")
            // Dot product between camera forward and direction to entity
            float focus = glm::dot(cameraForward, dirToEntity);
            if (focus < 0.0f) focus = 0.0f; // Only focus on what's in front

            // 3. Update Influence based on Focus and Distance
            // Higher focus and lower distance = higher influence
            float influence = (focus * focus) / (distance * 0.1f + 1.0f);
            phryll.observerInfluence = glm::mix(phryll.observerInfluence, influence, dt * 5.0f);

            // 4. Update Phryll Density (manifestation based on observation)
            // Scalar fields are affected by conscious observation (Focus)
            phryll.density = 0.5f + (phryll.observerInfluence * 0.5f);
            if (phryll.density > 1.0f) phryll.density = 1.0f;

            // 5. Update Frequency (Vibrational Shift)
            // Entities "vibrate higher" when observed in a scalar field
            phryll.currentFrequency = phryll.baseFrequency * (1.0f + phryll.observerInfluence * 0.2f);

            // 6. Manifestation Logic
            // If the entity has a TimeComponent, check if it's "in focus" enough to manifest
            if (registry.all_of<components::TimeComponent>(entity)) {
                auto& timeComp = registry.get<components::TimeComponent>(entity);
                // Lower density entities (e.g. 5th density) might only manifest if phryll density is high enough
                if (timeComp.density > 5.0f) {
                    phryll.isManifesting = (phryll.density > 0.6f);
                } else {
                    phryll.isManifesting = true; // Physical entities always manifest
                }
            }
        }
    }

    void ScalarFieldSystem::visualize(entt::registry& registry, renderer::DebugRenderer* debugRenderer) {
        // Not implemented yet, could draw glowing spheres around highly observed entities
    }

} // namespace Umgebung::ecs::systems