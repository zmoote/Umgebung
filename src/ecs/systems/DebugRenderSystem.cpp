#include "umgebung/ecs/systems/DebugRenderSystem.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::ecs::systems
{

    DebugRenderSystem::DebugRenderSystem(renderer::DebugRenderer* debugRenderer)
        : debugRenderer_(debugRenderer)
    {
    }

    void DebugRenderSystem::onUpdate(entt::registry& registry)
    {
        if (!enabled_ || !debugRenderer_) return;

        auto view = registry.view<components::Transform, components::Collider>();
        for (auto entity : view)
        {
            auto& transform = view.get<components::Transform>(entity);
            auto& collider = view.get<components::Collider>(entity);

            glm::vec4 color = { 0.0f, 1.0f, 0.0f, 1.0f }; // Green for static
            if (registry.all_of<components::RigidBody>(entity))
            {
                auto& rigidBody = registry.get<components::RigidBody>(entity);
                if (rigidBody.type == components::RigidBody::BodyType::Dynamic)
                {
                    color = { 1.0f, 0.0f, 0.0f, 1.0f }; // Red for dynamic
                }
            }
            
            glm::mat4 modelMatrixNoScale = glm::translate(glm::mat4(1.0f), transform.position) * glm::mat4_cast(transform.rotation);
            
            switch (collider.type)
            {
                case components::Collider::ColliderType::Box:
                {
                    glm::vec3 finalHalfExtents = collider.boxSize * transform.scale;
                    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), finalHalfExtents * 2.0f);
                    debugRenderer_->drawBox(modelMatrixNoScale * scaleMatrix, color);
                    break;
                }
                case components::Collider::ColliderType::Sphere:
                {
                    float maxScale = glm::max(transform.scale.x, glm::max(transform.scale.y, transform.scale.z));
                    float finalRadius = collider.sphereRadius * maxScale;
                    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(finalRadius * 2.0f));
                    debugRenderer_->drawSphere(modelMatrixNoScale * scaleMatrix, color);
                    break;
                }
            }
        }
    }

} // namespace Umgebung::ecs::systems
