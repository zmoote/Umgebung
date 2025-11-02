/**
 * @file Transform.cpp
 * @brief Implements the Transform class.
 */
#include "umgebung/ecs/components/Transform.hpp"

#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::ecs::components {

    glm::mat4 Transform::getModelMatrix() const {
        glm::mat4 model = glm::mat4(1.0f);

        model = glm::translate(model, position);

        model = model * glm::mat4_cast(rotation);

        model = glm::scale(model, scale);

        return model;
    }

}