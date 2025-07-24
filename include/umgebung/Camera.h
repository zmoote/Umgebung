#pragma once

#include <DirectXMath.h>
#include <memory>
#include <unordered_map>

namespace Umgebung {

    enum class CameraMode {
        Orthographic2D,
        Perspective3D
    };

    enum class CameraMovement {
        Forward,
        Backward,
        Left,
        Right,
        Up,
        Down
    };

    class Camera {
    public:
        Camera();
        ~Camera() = default;

        // Core camera functions
        void Update(float deltaTime);
        void SetViewportSize(int width, int height);

        // Mode switching
        void SetMode(CameraMode mode);
        CameraMode GetMode() const { return m_mode; }
        void ToggleMode();

        // Matrix getters
        DirectX::XMMATRIX GetViewMatrix() const;
        DirectX::XMMATRIX GetProjectionMatrix() const;
        DirectX::XMMATRIX GetViewProjectionMatrix() const;

        // 3D Camera controls
        void ProcessMouseMovement(float xOffset, float yOffset, bool constrainPitch = true);
        void ProcessMouseScroll(float yOffset);
        void ProcessKeyboard(CameraMovement direction, float deltaTime);

        // 2D Camera controls
        void Pan2D(float xOffset, float yOffset);
        void Zoom2D(float factor);
        void SetZoom2D(float zoom);

        // Position and orientation
        void SetPosition(const DirectX::XMFLOAT3& position);
        void SetTarget(const DirectX::XMFLOAT3& target);
        void LookAt(const DirectX::XMFLOAT3& position, const DirectX::XMFLOAT3& target, const DirectX::XMFLOAT3& up);

        DirectX::XMFLOAT3 GetPosition() const { return m_position; }
        DirectX::XMFLOAT3 GetFront() const { return m_front; }
        DirectX::XMFLOAT3 GetUp() const { return m_up; }
        DirectX::XMFLOAT3 GetRight() const { return m_right; }

        // Camera properties
        void SetMovementSpeed(float speed) { m_movementSpeed = speed; }
        void SetMouseSensitivity(float sensitivity) { m_mouseSensitivity = sensitivity; }
        void SetFieldOfView(float fov);
        void SetNearFar(float nearPlane, float farPlane);

        float GetMovementSpeed() const { return m_movementSpeed; }
        float GetMouseSensitivity() const { return m_mouseSensitivity; }
        float GetFieldOfView() const { return m_fov; }
        float GetZoom2D() const { return m_zoom2D; }

        // Orbital camera for 3D object inspection
        void SetOrbitalMode(bool enabled, const DirectX::XMFLOAT3& target = DirectX::XMFLOAT3(0, 0, 0));
        void OrbitAroundTarget(float yawDelta, float pitchDelta);
        void SetOrbitalDistance(float distance);

        // Smooth transitions
        void SmoothTransitionTo(const DirectX::XMFLOAT3& targetPosition, const DirectX::XMFLOAT3& targetLookAt, float duration);
        void FocusOnObject(const DirectX::XMFLOAT3& objectPosition, float objectRadius, float duration = 1.0f);

        // Reset functions
        void ResetToDefault();
        void Reset2DView();
        void Reset3DView();

        // Coordinate space conversions for multiverse navigation
        DirectX::XMFLOAT2 WorldToScreen(const DirectX::XMFLOAT3& worldPos) const;
        DirectX::XMFLOAT3 ScreenToWorld(const DirectX::XMFLOAT2& screenPos, float depth = 0.0f) const;

    private:
        // Camera state
        CameraMode m_mode;
        DirectX::XMFLOAT3 m_position;
        DirectX::XMFLOAT3 m_front;
        DirectX::XMFLOAT3 m_up;
        DirectX::XMFLOAT3 m_right;
        DirectX::XMFLOAT3 m_worldUp;

        // 3D camera angles
        float m_yaw;
        float m_pitch;

        // 2D camera state
        DirectX::XMFLOAT2 m_center2D;
        float m_zoom2D;
        float m_minZoom2D;
        float m_maxZoom2D;

        // Camera options
        float m_movementSpeed;
        float m_mouseSensitivity;
        float m_fov;
        float m_nearPlane;
        float m_farPlane;

        // Viewport
        int m_viewportWidth;
        int m_viewportHeight;
        float m_aspectRatio;

        // Orbital camera state
        bool m_orbitalMode;
        DirectX::XMFLOAT3 m_orbitalTarget;
        float m_orbitalDistance;
        float m_orbitalYaw;
        float m_orbitalPitch;

        // Smooth transition state
        bool m_inTransition;
        float m_transitionTimer;
        float m_transitionDuration;
        DirectX::XMFLOAT3 m_transitionStartPos;
        DirectX::XMFLOAT3 m_transitionEndPos;
        DirectX::XMFLOAT3 m_transitionStartLookAt;
        DirectX::XMFLOAT3 m_transitionEndLookAt;

        // Internal update functions
        void UpdateCameraVectors();
        void UpdateTransition(float deltaTime);
        DirectX::XMFLOAT3 Lerp(const DirectX::XMFLOAT3& a, const DirectX::XMFLOAT3& b, float t);
        float EaseInOutCubic(float t);

        // Default values
        static constexpr float DEFAULT_YAW = -90.0f;
        static constexpr float DEFAULT_PITCH = 0.0f;
        static constexpr float DEFAULT_SPEED = 2.5f;
        static constexpr float DEFAULT_SENSITIVITY = 0.1f;
        static constexpr float DEFAULT_FOV = 45.0f;
        static constexpr float DEFAULT_NEAR = 0.1f;
        static constexpr float DEFAULT_FAR = 1000.0f;
        static constexpr float DEFAULT_ZOOM_2D = 1.0f;
        static constexpr float MIN_ZOOM_2D = 0.001f;
        static constexpr float MAX_ZOOM_2D = 1000.0f;
    };

    // Utility class for camera presets and saved positions
    class CameraPresets {
    public:
        struct CameraState {
            CameraMode mode;
            DirectX::XMFLOAT3 position;
            DirectX::XMFLOAT3 target;
            float zoom2D;
            DirectX::XMFLOAT2 center2D;
            float yaw, pitch;
            std::string name;
        };

        void SavePreset(const std::string& name, const Camera& camera);
        bool LoadPreset(const std::string& name, Camera& camera);
        void DeletePreset(const std::string& name);
        std::vector<std::string> GetPresetNames() const;

        // Built-in presets for multiverse navigation
        void CreateDefaultPresets();
        CameraState GetMultiverseOverview() const;
        CameraState GetUniverseView() const;
        CameraState GetGalaxyView() const;
        CameraState GetStarSystemView() const;

    private:
        std::unordered_map<std::string, CameraState> m_presets;
    };

}