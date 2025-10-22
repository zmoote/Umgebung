#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nlohmann/json.hpp>
#include "umgebung/util/JsonHelpers.hpp"

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

            // --- ADD THIS MACRO ---
            // This tells nlohmann::json how to serialize/deserialize your struct
            // Make sure the names match your struct members exactly.
            NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TransformComponent, position, rotation, scale)
                // ---------------------

            }
        }
    }