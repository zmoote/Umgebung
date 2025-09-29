#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Umgebung {
    namespace ecs {
        namespace components {

            struct TransformComponent {
                
                glm::vec3 position{ 0.0f, 0.0f, 0.0f };

                glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };

                glm::vec3 scale{ 1.0f, 1.0f, 1.0f };

                TransformComponent() = default;

                TransformComponent(const glm::vec3& pos) : position(pos) {}

                glm::mat4 getModelMatrix() const;
            };

            }
        }
    }