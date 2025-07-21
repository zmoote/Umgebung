#ifndef UMBGEBUNG_CAMERA_HPP
#define UMBGEBUNG_CAMERA_HPP

#include <DirectXMath.h>

namespace Umgebung {

    class Camera {
    public:
        enum class ViewMode { View2D, View3D };  // Enum for 2D/3D modes

        Camera(float width, float height, float fov = DirectX::XM_PIDIV4, float nearZ = 0.1f, float farZ = 100.0f);
        void Update(float deltaTime, bool mouseDown, int mouseDeltaX, int mouseDeltaY);
        void SetViewMode(ViewMode mode);
        DirectX::XMMATRIX GetViewMatrix() const;
        DirectX::XMMATRIX GetProjectionMatrix() const;
        ViewMode GetViewMode() const;  // Declare the missing method

    private:
        void UpdateViewMatrix();
        void UpdateProjectionMatrix();

        DirectX::XMVECTOR position_, target_, up_;
        DirectX::XMMATRIX viewMatrix_, projMatrix_;
        float width_, height_, fov_, nearZ_, farZ_;
        ViewMode viewMode_;  // Member to store the current mode
        float yaw_, pitch_;
    };

} // namespace Umgebung

#endif // UMBGEBUNG_CAMERA_HPP