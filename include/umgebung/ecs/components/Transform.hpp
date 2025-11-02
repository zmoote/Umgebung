#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nlohmann/json.hpp>
#include "umgebung/util/JsonHelpers.hpp"

namespace Umgebung {
    namespace ecs {
        namespace components {

            /**
 * @file Transform.hpp
 * @brief Contains the Transform component.
 */
#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nlohmann/json.hpp>
#include "umgebung/util/JsonHelpers.hpp"

namespace Umgebung {
    namespace ecs {
        namespace components {

            /**
             * @brief A component that represents the position, rotation, and scale of an entity.
             */
            struct Transform {
                
                glm::vec3 position{ 0.0f, 0.0f, 0.0f }; ///< The position of the entity.

                glm::quat rotation{ 1.0f, 0.0f, 0.0f, 0.0f }; ///< The rotation of the entity.

                glm::vec3 scale{ 1.0f, 1.0f, 1.0f }; ///< The scale of the entity.

                /**
                 * @brief Default constructor.
                 */
                Transform() = default;

                /**
                 * @brief Construct a new Transform object with a given position.
                 * @param pos The position.
                 */
                Transform(const glm::vec3& pos) : position(pos) {}

                /**
                 * @brief Get the model matrix of the entity.
                 * @return The model matrix.
                 */
                glm::mat4 getModelMatrix() const;
            };

            /**
             * @brief This tells nlohmann::json how to serialize/deserialize your struct.
             * Make sure the names match your struct members exactly.
             */
            NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Transform, position, rotation, scale)

            }
        }
    }