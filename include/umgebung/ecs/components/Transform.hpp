#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace Umgebung {
    namespace ecs {
        namespace components {

            /**
             * @struct TransformComponent
             * @brief Holds position, rotation, and scale data for an entity.
             *
             * This component provides all the necessary information to place an object
             * in 3D world space. It uses a quaternion for rotation to avoid issues
             * like gimbal lock.
             */
            struct TransformComponent {
                // The world space position of the entity.
                glm::vec3 position{ 0.0f, 0.0f, 0.0f };

                // The orientation of the entity, represented by a quaternion.
                // Default is an identity quaternion (no rotation).
                glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f };

                // The scale of the entity along each axis (X, Y, Z).
                glm::vec3 scale{ 1.0f, 1.0f, 1.0f };

                /**
                 * @brief Default constructor. Initializes to an identity transform.
                 */
                TransformComponent() = default;

                /**
                 * @brief Constructs a TransformComponent with a specific position.
                 * @param pos The initial position of the entity.
                 */
                TransformComponent(const glm::vec3& pos) : position(pos) {}

                /**
                 * @brief Calculates and returns the model matrix for this transform.
                 *
                 * The model matrix transforms the entity's vertices from local space
                 * to world space. It combines the scale, rotation, and position.
                 * @return A 4x4 model matrix (glm::mat4).
                 */
                glm::mat4 getModelMatrix() const;
            };

            } // namespace components
        } // namespace ecs
    } // namespace Umgebung