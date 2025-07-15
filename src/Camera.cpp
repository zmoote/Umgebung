#include "../include/Camera.h"
#include <DirectXMath.h>
using namespace DirectX;

Camera::Camera() : m_position(0,0,0), m_pitch(0), m_yaw(0) {}

void Camera::SetPosition(const XMFLOAT3& pos) { m_position = pos; }
void Camera::SetRotation(float pitch, float yaw) { m_pitch = pitch; m_yaw = yaw; }
void Camera::Move(const XMFLOAT3& delta) {
    XMVECTOR forward = XMVectorSet(
        sinf(m_yaw), 0, cosf(m_yaw), 0
    );
    XMVECTOR right = XMVectorSet(
        cosf(m_yaw), 0, -sinf(m_yaw), 0
    );
    XMVECTOR up = XMVectorSet(0, 1, 0, 0);
    XMVECTOR d = XMVectorSet(delta.x, delta.y, delta.z, 0);
    XMVECTOR pos = XMLoadFloat3(&m_position);
    pos += XMVectorScale(forward, delta.z);
    pos += XMVectorScale(right, delta.x);
    pos += XMVectorScale(up, delta.y);
    XMStoreFloat3(&m_position, pos);
}
void Camera::Rotate(float dPitch, float dYaw) {
    m_pitch += dPitch;
    m_yaw += dYaw;
    if (m_pitch > XM_PIDIV2) m_pitch = XM_PIDIV2;
    if (m_pitch < -XM_PIDIV2) m_pitch = -XM_PIDIV2;
}
XMMATRIX Camera::GetViewMatrix() const {
    XMVECTOR pos = XMLoadFloat3(&m_position);
    float cp = cosf(m_pitch), sp = sinf(m_pitch), cy = cosf(m_yaw), sy = sinf(m_yaw);
    XMVECTOR look = XMVectorSet(cp*sy, sp, cp*cy, 0);
    XMVECTOR up = XMVectorSet(0, 1, 0, 0);
    return XMMatrixLookToLH(pos, look, up);
}
XMMATRIX Camera::GetProjectionMatrix(float aspect, float fov, float nearZ, float farZ) const {
    return XMMatrixPerspectiveFovLH(fov, aspect, nearZ, farZ);
}
const XMFLOAT3& Camera::GetPosition() const { return m_position; }
float Camera::GetPitch() const { return m_pitch; }
float Camera::GetYaw() const { return m_yaw; }
