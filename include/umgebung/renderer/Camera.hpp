#pragma once

#include <glm/glm.hpp>

namespace Umgebung::renderer {

    class Camera {
    public:
        // A simpler constructor for now
        Camera();

        // --- Methods our Renderer needs ---
        void setPerspective(float fov, float aspectRatio, float nearPlane, float farPlane);
        void setPosition(const glm::vec3& position);

        // --- Existing Getters ---
        const glm::mat4& getViewMatrix() const { return viewMatrix_; }
        const glm::mat4& getProjectionMatrix() const { return projectionMatrix_; }

    private:
        void updateViewMatrix();

        glm::mat4 projectionMatrix_{ 1.0f };
        glm::mat4 viewMatrix_{ 1.0f };

        glm::vec3 position_{ 0.0f, 0.0f, 3.0f };
        glm::vec3 front_{ 0.0f, 0.0f, -1.0f };
        glm::vec3 up_{ 0.0f, 1.0f, 0.0f };
    };

} // namespace Umgebung::renderer