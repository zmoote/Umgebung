/**
 * @file Camera.hpp
 * @brief Contains the Camera class.
 */
#pragma once

#include <glm/glm.hpp>

namespace Umgebung::renderer {

    /**
     * @brief An enum for camera movement directions.
     */
    enum class Camera_Movement {
        FORWARD,    ///< Move forward
        BACKWARD,   ///< Move backward
        LEFT,       ///< Move left
        RIGHT       ///< Move right
    };

    /**
     * @brief A class for managing the camera's position, orientation, and projection.
     */
    class Camera {
    public:
        /**
         * @brief Construct a new Camera object.
         */
        Camera();

        /**
         * @brief Construct a new Camera object with a specific position.
         * 
         * @param position The initial position.
         */
        Camera(const glm::vec3& position);

        /**
         * @brief Processes input received from any keyboard-like input system.
         * 
         * @param direction The direction to move the camera.
         * @param deltaTime The time since the last frame.
         */
        void processKeyboard(Camera_Movement direction, float deltaTime);

        /**
         * @brief Processes input received from a mouse input system.
         * 
         * @param xoffset The mouse's x-offset.
         * @param yoffset The mouse's y-offset.
         * @param constrainPitch Whether to constrain the pitch.
         */
        void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

        /**
         * @brief Set the Perspective object.
         * 
         * @param fov The field of view.
         * @param aspectRatio The aspect ratio.
         * @param nearPlane The near plane.
         * @param farPlane The far plane.
         */
        void setPerspective(float fov, float aspectRatio, float nearPlane, float farPlane);

        /**
         * @brief Sets the near and far clipping planes.
         * @param nearPlane The new near plane.
         * @param farPlane The new far plane.
         */
        void setPlanes(float nearPlane, float farPlane);

        /**
         * @brief Set the Position object.
         * 
         * @param position The new position.
         */
        void setPosition(const glm::vec3& position);

        /**
         * @brief Get the Position object.
         * 
         * @return const glm::vec3& 
         */
        const glm::vec3& getPosition() const { return position_; }

        float getYaw() const { return yaw_; }
        float getPitch() const { return pitch_; }
        void setYaw(float yaw);
        void setPitch(float pitch);

        /**
         * @brief Get the View Matrix object.
         * 
         * @return const glm::mat4& 
         */
        const glm::mat4& getViewMatrix() const { return viewMatrix_; }

        /**
         * @brief Get the Projection Matrix object.
         * 
         * @return const glm::mat4& 
         */
        const glm::mat4& getProjectionMatrix() const { return projectionMatrix_; }

    private:
        /**
         * @brief Updates the camera's vectors.
         */
        void updateCameraVectors();

        glm::mat4 projectionMatrix_{ 1.0f }; ///< The projection matrix.
        glm::mat4 viewMatrix_{ 1.0f };       ///< The view matrix.

        glm::vec3 position_{ 0.0f, 0.0f, 3.0f }; ///< The camera's position.
        glm::vec3 front_{ 0.0f, 0.0f, -1.0f };    ///< The camera's front vector.
        glm::vec3 up_{ 0.0f, 1.0f, 0.0f };        ///< The camera's up vector.
        glm::vec3 right_{ 1.0f, 0.0f, 0.0f };     ///< The camera's right vector.
        glm::vec3 worldUp_{ 0.0f, 1.0f, 0.0f };   ///< The world's up vector.

        float yaw_ = -90.0f;                ///< The camera's yaw.
        float pitch_ = 0.0f;                 ///< The camera's pitch.

        float movementSpeed_ = 2.5f;         ///< The camera's movement speed.
        float mouseSensitivity_ = 0.1f;      ///< The camera's mouse sensitivity.

        // Perspective settings
        float fov_ = 45.0f;
        float aspectRatio_ = 1.0f;
        float nearPlane_ = 0.1f;
        float farPlane_ = 100.0f;
    };

}