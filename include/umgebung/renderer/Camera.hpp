#pragma once

#include <glm/glm.hpp>

namespace Umgebung::renderer {

    enum class Camera_Movement {
        FORWARD,
        BACKWARD,
        LEFT,
        RIGHT
    };

    class Camera {
    public:
        Camera();

        void processKeyboard(Camera_Movement direction, float deltaTime);
        void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

        void setPerspective(float fov, float aspectRatio, float nearPlane, float farPlane);
        void setPosition(const glm::vec3& position);
        const glm::mat4& getViewMatrix() const { return viewMatrix_; }
        const glm::mat4& getProjectionMatrix() const { return projectionMatrix_; }

    private:
        void updateCameraVectors();

        glm::mat4 projectionMatrix_{ 1.0f };
        glm::mat4 viewMatrix_{ 1.0f };

        glm::vec3 position_{ 0.0f, 0.0f, 3.0f };
        glm::vec3 front_{ 0.0f, 0.0f, -1.0f };
        glm::vec3 up_{ 0.0f, 1.0f, 0.0f };
        glm::vec3 right_{ 1.0f, 0.0f, 0.0f };
        glm::vec3 worldUp_{ 0.0f, 1.0f, 0.0f };

        float yaw_ = -90.0f;
        float pitch_ = 0.0f;

        float movementSpeed_ = 2.5f;
        float mouseSensitivity_ = 0.1f;
    };

}