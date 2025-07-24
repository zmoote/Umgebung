#include "../../include/umgebung/Camera.h"
#include <algorithm>
#include <cmath>

using namespace DirectX;

namespace Umgebung {

    Camera::Camera()
        : m_mode(CameraMode::Perspective3D)
        , m_position(0.0f, 0.0f, 3.0f)
        , m_front(0.0f, 0.0f, -1.0f)
        , m_up(0.0f, 1.0f, 0.0f)
        , m_right(1.0f, 0.0f, 0.0f)
        , m_worldUp(0.0f, 1.0f, 0.0f)
        , m_yaw(DEFAULT_YAW)
        , m_pitch(DEFAULT_PITCH)
        , m_center2D(0.0f, 0.0f)
        , m_zoom2D(DEFAULT_ZOOM_2D)
        , m_minZoom2D(MIN_ZOOM_2D)
        , m_maxZoom2D(MAX_ZOOM_2D)
        , m_movementSpeed(DEFAULT_SPEED)
        , m_mouseSensitivity(DEFAULT_SENSITIVITY)
        , m_fov(DEFAULT_FOV)
        , m_nearPlane(DEFAULT_NEAR)
        , m_farPlane(DEFAULT_FAR)
        , m_viewportWidth(1280)
        , m_viewportHeight(720)
        , m_aspectRatio(16.0f / 9.0f)
        , m_orbitalMode(false)
        , m_orbitalTarget(0.0f, 0.0f, 0.0f)
        , m_orbitalDistance(5.0f)
        , m_orbitalYaw(0.0f)
        , m_orbitalPitch(0.0f)
        , m_inTransition(false)
        , m_transitionTimer(0.0f)
        , m_transitionDuration(0.0f)
        , m_transitionStartPos(0.0f, 0.0f, 0.0f)
        , m_transitionEndPos(0.0f, 0.0f, 0.0f)
        , m_transitionStartLookAt(0.0f, 0.0f, 0.0f)
        , m_transitionEndLookAt(0.0f, 0.0f, 0.0f)
    {
        UpdateCameraVectors();
    }

    void Camera::Update(float deltaTime) {
        if (m_inTransition) {
            UpdateTransition(deltaTime);
        }

        if (m_orbitalMode && m_mode == CameraMode::Perspective3D) {
            // Update orbital camera position
            float x = m_orbitalTarget.x + m_orbitalDistance * cosf(XMConvertToRadians(m_orbitalPitch)) * cosf(XMConvertToRadians(m_orbitalYaw));
            float y = m_orbitalTarget.y + m_orbitalDistance * sinf(XMConvertToRadians(m_orbitalPitch));
            float z = m_orbitalTarget.z + m_orbitalDistance * cosf(XMConvertToRadians(m_orbitalPitch)) * sinf(XMConvertToRadians(m_orbitalYaw));

            m_position = XMFLOAT3(x, y, z);

            // Look at the target
            XMVECTOR pos = XMLoadFloat3(&m_position);
            XMVECTOR target = XMLoadFloat3(&m_orbitalTarget);
            XMVECTOR up = XMLoadFloat3(&m_worldUp);
            XMVECTOR front = XMVector3Normalize(XMVectorSubtract(target, pos));

            XMStoreFloat3(&m_front, front);
            UpdateCameraVectors();
        }
    }

    void Camera::SetViewportSize(int width, int height) {
        m_viewportWidth = width;
        m_viewportHeight = height;
        m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    }

    void Camera::SetMode(CameraMode mode) {
        if (m_mode != mode) {
            m_mode = mode;

            // Smooth transition between modes
            if (mode == CameraMode::Orthographic2D) {
                // When switching to 2D, center the view on current looking direction
                XMVECTOR front = XMLoadFloat3(&m_front);
                XMVECTOR pos = XMLoadFloat3(&m_position);
                XMVECTOR target = XMVectorAdd(pos, front);

                XMFLOAT3 targetPos;
                XMStoreFloat3(&targetPos, target);
                m_center2D = XMFLOAT2(targetPos.x, targetPos.z); // Use X-Z plane for 2D
            }
        }
    }

    void Camera::ToggleMode() {
        SetMode(m_mode == CameraMode::Orthographic2D ? CameraMode::Perspective3D : CameraMode::Orthographic2D);
    }

    XMMATRIX Camera::GetViewMatrix() const {
        if (m_mode == CameraMode::Orthographic2D) {
            // 2D top-down view
            XMVECTOR eye = XMVectorSet(m_center2D.x, 100.0f, m_center2D.y, 1.0f);
            XMVECTOR at = XMVectorSet(m_center2D.x, 0.0f, m_center2D.y, 1.0f);
            XMVECTOR up = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f); // Z-up for 2D view
            return XMMatrixLookAtLH(eye, at, up);
        }
        else {
            // 3D perspective view
            XMVECTOR pos = XMLoadFloat3(&m_position);
            XMVECTOR front = XMLoadFloat3(&m_front);
            XMVECTOR up = XMLoadFloat3(&m_up);
            XMVECTOR target = XMVectorAdd(pos, front);
            return XMMatrixLookAtLH(pos, target, up);
        }
    }

    XMMATRIX Camera::GetProjectionMatrix() const {
        if (m_mode == CameraMode::Orthographic2D) {
            // Orthographic projection for 2D view
            float width = 20.0f / m_zoom2D;  // Adjustable viewport size
            float height = width / m_aspectRatio;
            return XMMatrixOrthographicLH(width, height, -1000.0f, 1000.0f);
        }
        else {
            // Perspective projection for 3D view
            return XMMatrixPerspectiveFovLH(XMConvertToRadians(m_fov), m_aspectRatio, m_nearPlane, m_farPlane);
        }
    }

    XMMATRIX Camera::GetViewProjectionMatrix() const {
        return XMMatrixMultiply(GetViewMatrix(), GetProjectionMatrix());
    }

    void Camera::ProcessMouseMovement(float xOffset, float yOffset, bool constrainPitch) {
        if (m_mode == CameraMode::Perspective3D && !m_orbitalMode) {
            xOffset *= m_mouseSensitivity;
            yOffset *= m_mouseSensitivity;

            m_yaw += xOffset;
            m_pitch += yOffset;

            if (constrainPitch) {
                m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);
            }

            UpdateCameraVectors();
        }
    }

    void Camera::ProcessMouseScroll(float yOffset) {
        if (m_mode == CameraMode::Orthographic2D) {
            Zoom2D(yOffset > 0 ? 1.1f : 0.9f);
        }
        else {
            // In 3D mode, scroll adjusts FOV
            m_fov -= yOffset;
            m_fov = std::clamp(m_fov, 1.0f, 120.0f);
        }
    }

    void Camera::ProcessKeyboard(CameraMovement direction, float deltaTime) {
        if (m_mode == CameraMode::Perspective3D && !m_orbitalMode) {
            float velocity = m_movementSpeed * deltaTime;

            XMVECTOR pos = XMLoadFloat3(&m_position);
            XMVECTOR front = XMLoadFloat3(&m_front);
            XMVECTOR up = XMLoadFloat3(&m_up);
            XMVECTOR right = XMLoadFloat3(&m_right);

            switch (direction) {
            case CameraMovement::Forward:
                pos = XMVectorAdd(pos, XMVectorScale(front, velocity));
                break;
            case CameraMovement::Backward:
                pos = XMVectorSubtract(pos, XMVectorScale(front, velocity));
                break;
            case CameraMovement::Left:
                pos = XMVectorSubtract(pos, XMVectorScale(right, velocity));
                break;
            case CameraMovement::Right:
                pos = XMVectorAdd(pos, XMVectorScale(right, velocity));
                break;
            case CameraMovement::Up:
                pos = XMVectorAdd(pos, XMVectorScale(up, velocity));
                break;
            case CameraMovement::Down:
                pos = XMVectorSubtract(pos, XMVectorScale(up, velocity));
                break;
            }

            XMStoreFloat3(&m_position, pos);
        }
    }

    void Camera::Pan2D(float xOffset, float yOffset) {
        if (m_mode == CameraMode::Orthographic2D) {
            float panSpeed = 1.0f / m_zoom2D;
            m_center2D.x -= xOffset * panSpeed;
            m_center2D.y -= yOffset * panSpeed;
        }
    }

    void Camera::Zoom2D(float factor) {
        if (m_mode == CameraMode::Orthographic2D) {
            m_zoom2D *= factor;
            m_zoom2D = std::clamp(m_zoom2D, m_minZoom2D, m_maxZoom2D);
        }
    }

    void Camera::SetZoom2D(float zoom) {
        m_zoom2D = std::clamp(zoom, m_minZoom2D, m_maxZoom2D);
    }

    void Camera::SetPosition(const XMFLOAT3& position) {
        m_position = position;
    }

    void Camera::SetTarget(const XMFLOAT3& target) {
        XMVECTOR pos = XMLoadFloat3(&m_position);
        XMVECTOR tgt = XMLoadFloat3(&target);
        XMVECTOR front = XMVector3Normalize(XMVectorSubtract(tgt, pos));
        XMStoreFloat3(&m_front, front);
        UpdateCameraVectors();
    }

    void Camera::LookAt(const XMFLOAT3& position, const XMFLOAT3& target, const XMFLOAT3& up) {
        m_position = position;
        m_worldUp = up;

        XMVECTOR pos = XMLoadFloat3(&position);
        XMVECTOR tgt = XMLoadFloat3(&target);
        XMVECTOR front = XMVector3Normalize(XMVectorSubtract(tgt, pos));
        XMStoreFloat3(&m_front, front);

        UpdateCameraVectors();
    }

    void Camera::SetFieldOfView(float fov) {
        m_fov = std::clamp(fov, 1.0f, 120.0f);
    }

    void Camera::SetNearFar(float nearPlane, float farPlane) {
        m_nearPlane = nearPlane;
        m_farPlane = farPlane;
    }

    void Camera::SetOrbitalMode(bool enabled, const XMFLOAT3& target) {
        m_orbitalMode = enabled;
        m_orbitalTarget = target;

        if (enabled) {
            // Calculate initial orbital angles based on current position
            XMVECTOR pos = XMLoadFloat3(&m_position);
            XMVECTOR tgt = XMLoadFloat3(&target);
            XMVECTOR diff = XMVectorSubtract(pos, tgt);

            XMFLOAT3 diffFloat;
            XMStoreFloat3(&diffFloat, diff);

            m_orbitalDistance = XMVectorGetX(XMVector3Length(diff));
            m_orbitalYaw = XMConvertToDegrees(atan2f(diffFloat.z, diffFloat.x));
            m_orbitalPitch = XMConvertToDegrees(asinf(diffFloat.y / m_orbitalDistance));
        }
    }

    void Camera::OrbitAroundTarget(float yawDelta, float pitchDelta) {
        if (m_orbitalMode) {
            m_orbitalYaw += yawDelta * m_mouseSensitivity;
            m_orbitalPitch += pitchDelta * m_mouseSensitivity;
            m_orbitalPitch = std::clamp(m_orbitalPitch, -89.0f, 89.0f);
        }
    }

    void Camera::SetOrbitalDistance(float distance) {
        m_orbitalDistance = std::max(0.1f, distance);
    }

    void Camera::SmoothTransitionTo(const XMFLOAT3& targetPosition, const XMFLOAT3& targetLookAt, float duration) {
        m_inTransition = true;
        m_transitionTimer = 0.0f;
        m_transitionDuration = duration;
        m_transitionStartPos = m_position;
        m_transitionEndPos = targetPosition;

        // Calculate current look-at point
        XMVECTOR pos = XMLoadFloat3(&m_position);
        XMVECTOR front = XMLoadFloat3(&m_front);
        XMVECTOR currentLookAt = XMVectorAdd(pos, front);
        XMStoreFloat3(&m_transitionStartLookAt, currentLookAt);
        m_transitionEndLookAt = targetLookAt;
    }

    void Camera::FocusOnObject(const XMFLOAT3& objectPosition, float objectRadius, float duration) {
        // Calculate optimal viewing distance based on object radius and FOV
        float distance = objectRadius / tanf(XMConvertToRadians(m_fov * 0.5f)) * 2.0f;
        distance = std::max(distance, objectRadius * 3.0f); // Minimum distance

        XMFLOAT3 targetPosition = XMFLOAT3(
            objectPosition.x + distance,
            objectPosition.y + distance * 0.3f,
            objectPosition.z + distance
        );

        SmoothTransitionTo(targetPosition, objectPosition, duration);
    }

    void Camera::ResetToDefault() {
        if (m_mode == CameraMode::Orthographic2D) {
            Reset2DView();
        }
        else {
            Reset3DView();
        }
    }

    void Camera::Reset2DView() {
        m_center2D = XMFLOAT2(0.0f, 0.0f);
        m_zoom2D = DEFAULT_ZOOM_2D;
    }

    void Camera::Reset3DView() {
        m_position = XMFLOAT3(0.0f, 0.0f, 3.0f);
        m_yaw = DEFAULT_YAW;
        m_pitch = DEFAULT_PITCH;
        m_fov = DEFAULT_FOV;
        m_orbitalMode = false;
        UpdateCameraVectors();
    }

    XMFLOAT2 Camera::WorldToScreen(const XMFLOAT3& worldPos) const {
        XMVECTOR world = XMLoadFloat3(&worldPos);
        XMMATRIX viewProj = GetViewProjectionMatrix();

        XMVECTOR screenPos = XMVector3TransformCoord(world, viewProj);

        XMFLOAT3 screenFloat;
        XMStoreFloat3(&screenFloat, screenPos);

        // Convert from NDC to screen coordinates
        float screenX = (screenFloat.x + 1.0f) * 0.5f * m_viewportWidth;
        float screenY = (1.0f - screenFloat.y) * 0.5f * m_viewportHeight;

        return XMFLOAT2(screenX, screenY);
    }

    XMFLOAT3 Camera::ScreenToWorld(const XMFLOAT2& screenPos, float depth) const {
        // Convert screen coordinates to NDC
        float ndcX = (screenPos.x / m_viewportWidth) * 2.0f - 1.0f;
        float ndcY = 1.0f - (screenPos.y / m_viewportHeight) * 2.0f;

        XMVECTOR screenVector = XMVectorSet(ndcX, ndcY, depth, 1.0f);

        XMMATRIX invViewProj = XMMatrixInverse(nullptr, GetViewProjectionMatrix());
        XMVECTOR worldPos = XMVector3TransformCoord(screenVector, invViewProj);

        XMFLOAT3 result;
        XMStoreFloat3(&result, worldPos);
        return result;
    }

    void Camera::UpdateCameraVectors() {
        if (m_mode == CameraMode::Perspective3D && !m_orbitalMode) {
            // Calculate front vector
            XMFLOAT3 front;
            front.x = cosf(XMConvertToRadians(m_yaw)) * cosf(XMConvertToRadians(m_pitch));
            front.y = sinf(XMConvertToRadians(m_pitch));
            front.z = sinf(XMConvertToRadians(m_yaw)) * cosf(XMConvertToRadians(m_pitch));

            XMVECTOR frontVec = XMVector3Normalize(XMLoadFloat3(&front));
            XMStoreFloat3(&m_front, frontVec);

            // Calculate right and up vectors
            XMVECTOR worldUp = XMLoadFloat3(&m_worldUp);
            XMVECTOR rightVec = XMVector3Normalize(XMVector3Cross(frontVec, worldUp));
            XMVECTOR upVec = XMVector3Normalize(XMVector3Cross(rightVec, frontVec));

            XMStoreFloat3(&m_right, rightVec);
            XMStoreFloat3(&m_up, upVec);
        }
    }

    void Camera::UpdateTransition(float deltaTime) {
        m_transitionTimer += deltaTime;
        float t = m_transitionTimer / m_transitionDuration;

        if (t >= 1.0f) {
            t = 1.0f;
            m_inTransition = false;
        }

        // Apply easing function
        float easedT = EaseInOutCubic(t);

        // Interpolate position
        m_position = Lerp(m_transitionStartPos, m_transitionEndPos, easedT);

        // Interpolate look-at target and update front vector
        XMFLOAT3 currentLookAt = Lerp(m_transitionStartLookAt, m_transitionEndLookAt, easedT);

        XMVECTOR pos = XMLoadFloat3(&m_position);
        XMVECTOR lookAt = XMLoadFloat3(&currentLookAt);
        XMVECTOR front = XMVector3Normalize(XMVectorSubtract(lookAt, pos));
        XMStoreFloat3(&m_front, front);

        UpdateCameraVectors();
    }

    XMFLOAT3 Camera::Lerp(const XMFLOAT3& a, const XMFLOAT3& b, float t) {
        return XMFLOAT3(
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z)
        );
    }

    float Camera::EaseInOutCubic(float t) {
        return t < 0.5f ? 4.0f * t * t * t : 1.0f - powf(-2.0f * t + 2.0f, 3.0f) / 2.0f;
    }

    // CameraPresets Implementation
    void CameraPresets::SavePreset(const std::string& name, const Camera& camera) {
        CameraState state;
        state.mode = camera.GetMode();
        state.position = camera.GetPosition();
        state.zoom2D = camera.GetZoom2D();
        // Note: You might need to add getters for center2D, yaw, pitch to Camera class
        state.name = name;

        m_presets[name] = state;
    }

    bool CameraPresets::LoadPreset(const std::string& name, Camera& camera) {
        auto it = m_presets.find(name);
        if (it == m_presets.end()) {
            return false;
        }

        const CameraState& state = it->second;
        camera.SetMode(state.mode);
        camera.SetPosition(state.position);

        if (state.mode == CameraMode::Orthographic2D) {
            camera.SetZoom2D(state.zoom2D);
        }

        return true;
    }

    void CameraPresets::DeletePreset(const std::string& name) {
        m_presets.erase(name);
    }

    std::vector<std::string> CameraPresets::GetPresetNames() const {
        std::vector<std::string> names;
        names.reserve(m_presets.size());

        for (const auto& pair : m_presets) {
            names.push_back(pair.first);
        }

        return names;
    }

    void CameraPresets::CreateDefaultPresets() {
        // Multiverse overview - very far out 2D view
        CameraState multiverseView;
        multiverseView.mode = CameraMode::Orthographic2D;
        multiverseView.position = XMFLOAT3(0.0f, 1000.0f, 0.0f);
        multiverseView.center2D = XMFLOAT2(0.0f, 0.0f);
        multiverseView.zoom2D = 0.001f;
        multiverseView.name = "Multiverse Overview";
        m_presets["multiverse"] = multiverseView;

        // Universe view - medium distance
        CameraState universeView;
        universeView.mode = CameraMode::Orthographic2D;
        universeView.position = XMFLOAT3(0.0f, 100.0f, 0.0f);
        universeView.center2D = XMFLOAT2(0.0f, 0.0f);
        universeView.zoom2D = 0.1f;
        universeView.name = "Universe View";
        m_presets["universe"] = universeView;

        // Galaxy view - closer 2D view
        CameraState galaxyView;
        galaxyView.mode = CameraMode::Orthographic2D;
        galaxyView.position = XMFLOAT3(0.0f, 50.0f, 0.0f);
        galaxyView.center2D = XMFLOAT2(0.0f, 0.0f);
        galaxyView.zoom2D = 1.0f;
        galaxyView.name = "Galaxy View";
        m_presets["galaxy"] = galaxyView;

        // Star system view - 3D perspective
        CameraState starSystemView;
        starSystemView.mode = CameraMode::Perspective3D;
        starSystemView.position = XMFLOAT3(10.0f, 5.0f, 10.0f);
        starSystemView.target = XMFLOAT3(0.0f, 0.0f, 0.0f);
        starSystemView.name = "Star System View";
        m_presets["starsystem"] = starSystemView;
    }

    CameraPresets::CameraState CameraPresets::GetMultiverseOverview() const {
        auto it = m_presets.find("multiverse");
        return (it != m_presets.end()) ? it->second : CameraState{};
    }

    CameraPresets::CameraState CameraPresets::GetUniverseView() const {
        auto it = m_presets.find("universe");
        return (it != m_presets.end()) ? it->second : CameraState{};
    }

    CameraPresets::CameraState CameraPresets::GetGalaxyView() const {
        auto it = m_presets.find("galaxy");
        return (it != m_presets.end()) ? it->second : CameraState{};
    }

    CameraPresets::CameraState CameraPresets::GetStarSystemView() const {
        auto it = m_presets.find("starsystem");
        return (it != m_presets.end()) ? it->second : CameraState{};
    }

}