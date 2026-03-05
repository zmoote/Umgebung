/**
 * @file Transform.cpp
 * @brief Implements the Transform class.
 */
#include "umgebung/ecs/components/Transform.hpp"

#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::ecs::components {

    glm::mat4 Transform::getModelMatrix() const {
        // Only recalculate if something has changed or if it's the first time
        if (isDirty || position != lastPosition_ || rotation != lastRotation_ || scale != lastScale_) {
            cachedModelMatrix_ = glm::mat4(1.0f);
            cachedModelMatrix_ = glm::translate(cachedModelMatrix_, position);
            cachedModelMatrix_ = cachedModelMatrix_ * glm::mat4_cast(rotation);
            cachedModelMatrix_ = glm::scale(cachedModelMatrix_, scale);

            lastPosition_ = position;
            lastRotation_ = rotation;
            lastScale_ = scale;
            isDirty = false;
        }

        return cachedModelMatrix_;
    }

}