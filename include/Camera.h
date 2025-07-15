#pragma once
#include <DirectXMath.h>

class Camera {
public:
    Camera();
    void SetPosition(const DirectX::XMFLOAT3& pos);
    void SetRotation(float pitch, float yaw);
    void Move(const DirectX::XMFLOAT3& delta);
    void Rotate(float dPitch, float dYaw);
    DirectX::XMMATRIX GetViewMatrix() const;
    DirectX::XMMATRIX GetProjectionMatrix(float aspect, float fov = DirectX::XM_PIDIV4, float nearZ = 0.1f, float farZ = 1000.0f) const;
    const DirectX::XMFLOAT3& GetPosition() const;
    float GetPitch() const;
    float GetYaw() const;
private:
    DirectX::XMFLOAT3 m_position;
    float m_pitch;
    float m_yaw;
};
