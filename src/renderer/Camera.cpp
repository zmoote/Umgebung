#include "umgebung/renderer/Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::renderer {

    Camera::Camera() {
        updateCameraVectors();
    }

    void Camera::processKeyboard(Camera_Movement direction, float deltaTime) {
        float velocity = movementSpeed_ * deltaTime;
        if (direction == Camera_Movement::FORWARD)
            position_ += front_ * velocity;
        if (direction == Camera_Movement::BACKWARD)
            position_ -= front_ * velocity;
        if (direction == Camera_Movement::LEFT)
            position_ -= right_ * velocity;
        if (direction == Camera_Movement::RIGHT)
            position_ += right_ * velocity;

        updateCameraVectors();
    }

    void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
        xoffset *= mouseSensitivity_;
        yoffset *= mouseSensitivity_;

        yaw_ += xoffset;
        pitch_ += yoffset;

        if (constrainPitch) {
            if (pitch_ > 89.0f)
                pitch_ = 89.0f;
            if (pitch_ < -89.0f)
                pitch_ = -89.0f;
        }

        updateCameraVectors();
    }

    void Camera::setPerspective(float fov, float aspectRatio, float nearPlane, float farPlane) {
        projectionMatrix_ = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
    }

    void Camera::setPosition(const glm::vec3& position) {
        position_ = position;
        updateCameraVectors();
    }

    void Camera::updateCameraVectors() {
        glm::vec3 front;
        front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front.y = sin(glm::radians(pitch_));
        front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
        front_ = glm::normalize(front);

        right_ = glm::normalize(glm::cross(front_, worldUp_));
        up_ = glm::normalize(glm::cross(right_, front_));

        viewMatrix_ = glm::lookAt(position_, position_ + front_, up_);
    }

}