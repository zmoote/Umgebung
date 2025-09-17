#include "umgebung/renderer/Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::renderer {

    Camera::Camera() {
        // Calculate the initial view matrix when the camera is created
        updateViewMatrix();
    }

    void Camera::setPerspective(float fov, float aspectRatio, float nearPlane, float farPlane) {
        projectionMatrix_ = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
    }

    void Camera::setPosition(const glm::vec3& position) {
        position_ = position;
        updateViewMatrix(); // Recalculate the view matrix whenever the position changes
    }

    void Camera::updateViewMatrix() {
        // A simple "lookAt" calculation
        viewMatrix_ = glm::lookAt(position_, position_ + front_, up_);
    }

} // namespace Umgebung::renderer