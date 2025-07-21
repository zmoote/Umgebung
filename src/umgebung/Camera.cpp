#include "../../include/umgebung/Camera.hpp"
#include <windows.h>
#include <algorithm>

namespace Umgebung {

    Camera::Camera(float width, float height, float fov, float nearZ, float farZ)
        : position_(DirectX::XMVectorSet(0.0f, 0.0f, -5.0f, 0.0f))
        , target_(DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f))
        , up_(DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f))
        , width_(width)
        , height_(height)
        , fov_(fov)
        , nearZ_(nearZ)
        , farZ_(farZ)
        , viewMode_(ViewMode::View3D)  // Default to 3D
        , yaw_(0.0f)
        , pitch_(0.0f)
    {
        UpdateViewMatrix();
        UpdateProjectionMatrix();
    }

    void Camera::Update(float deltaTime, bool mouseDown, int mouseDeltaX, int mouseDeltaY) {
        float speed = 5.0f * deltaTime;
        if (GetAsyncKeyState('W') & 0x8000) position_ = DirectX::XMVectorAdd(position_, DirectX::XMVectorScale(DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(target_, position_)), speed));
        if (GetAsyncKeyState('S') & 0x8000) position_ = DirectX::XMVectorSubtract(position_, DirectX::XMVectorScale(DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(target_, position_)), speed));
        if (GetAsyncKeyState('A') & 0x8000) position_ = DirectX::XMVectorSubtract(position_, DirectX::XMVectorScale(DirectX::XMVector3Cross(DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(target_, position_)), up_), speed));
        if (GetAsyncKeyState('D') & 0x8000) position_ = DirectX::XMVectorAdd(position_, DirectX::XMVectorScale(DirectX::XMVector3Cross(DirectX::XMVector3Normalize(DirectX::XMVectorSubtract(target_, position_)), up_), speed));

        if (viewMode_ == ViewMode::View3D && mouseDown) {
            yaw_ += mouseDeltaX * 0.005f;
            pitch_ = std::clamp(pitch_ + mouseDeltaY * 0.005f, -DirectX::XM_PIDIV2 + 0.1f, DirectX::XM_PIDIV2 - 0.1f);
            DirectX::XMVECTOR forward = DirectX::XMVectorSet(sinf(yaw_) * cosf(pitch_), sinf(pitch_), cosf(yaw_) * cosf(pitch_), 0.0f);
            target_ = DirectX::XMVectorAdd(position_, forward);
        }

        UpdateViewMatrix();
    }

    void Camera::SetViewMode(ViewMode mode) {
        viewMode_ = mode;
        UpdateProjectionMatrix();
    }

    DirectX::XMMATRIX Camera::GetViewMatrix() const {
        return viewMatrix_;
    }

    DirectX::XMMATRIX Camera::GetProjectionMatrix() const {
        return projMatrix_;
    }

    Camera::ViewMode Camera::GetViewMode() const {
        return viewMode_;  // Return the current view mode
    }

    void Camera::UpdateViewMatrix() {
        viewMatrix_ = DirectX::XMMatrixLookAtLH(position_, target_, up_);
    }

    void Camera::UpdateProjectionMatrix() {
        if (viewMode_ == ViewMode::View2D) {
            projMatrix_ = DirectX::XMMatrixOrthographicLH(width_, height_, nearZ_, farZ_);
        }
        else {
            projMatrix_ = DirectX::XMMatrixPerspectiveFovLH(fov_, width_ / height_, nearZ_, farZ_);
        }
    }

} // namespace Umgebung