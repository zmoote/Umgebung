#include "umgebung/ecs/components/Transform.hpp"

// GLM headers for matrix transformations
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::ecs::components {

    glm::mat4 TransformComponent::getModelMatrix() const {
        // 1. Start with an identity matrix.
        glm::mat4 model = glm::mat4(1.0f);

        // 2. Apply transformations in Scale -> Rotate -> Translate order.
        // This is the standard order to ensure transformations behave as expected.
        // For example, scaling should happen before translation, so the object
        // scales around its own origin, not the world origin.

        // Translate the model to its world position.
        model = glm::translate(model, position);

        // Rotate the model according to its quaternion orientation.
        // We convert the quaternion to a 4x4 rotation matrix.
        model = model * glm::mat4_cast(rotation);

        // Scale the model.
        model = glm::scale(model, scale);

        return model;
    }

} // namespace Umgebung::ecs::components