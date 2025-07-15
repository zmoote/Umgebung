//
// Game.h
//

#pragma once

#include "DeviceResources.h"
#include "StepTimer.h"
#include "Camera.h"

#include <memory>
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>


// A basic game implementation that creates a D3D12 device and
// provides a game loop.
class Game final : public DX::IDeviceNotify
{
public:

    Game() noexcept(false);
    ~Game();

    Game(Game&&) = default;
    Game& operator= (Game&&) = default;

    Game(Game const&) = delete;
    Game& operator= (Game const&) = delete;

    // Initialization and management
    void Initialize(HWND window, int width, int height);

    // Basic game loop
    void Tick();

    // IDeviceNotify
    void OnDeviceLost() override;
    void OnDeviceRestored() override;

    // Messages
    void OnActivated();
    void OnDeactivated();
    void OnSuspending();
    void OnResuming();
    void OnWindowMoved();
    void OnDisplayChange();
    void OnWindowSizeChanged(int width, int height);

    // Properties
    void GetDefaultSize( int& width, int& height ) const noexcept;

    // Camera input
    Camera& GetCamera();
    void OnCameraInput(float dx, float dy, bool mouse, float dt);

private:

    void Update(DX::StepTimer const& timer);
    void Render();

    void Clear();

    void CreateDeviceDependentResources();
    void CreateWindowSizeDependentResources();

    // Device resources.
    std::unique_ptr<DX::DeviceResources>        m_deviceResources;

    // Rendering loop timer.
    DX::StepTimer                               m_timer;

    // If using the DirectX Tool Kit for DX12, uncomment this line:
    std::unique_ptr<DirectX::GraphicsMemory> m_graphicsMemory;

    Camera m_camera;

    // Cube rendering resources
    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_cubeRootSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_cubePipelineState;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cubeVertexBuffer;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cubeIndexBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_cubeVBV = {};
    D3D12_INDEX_BUFFER_VIEW m_cubeIBV = {};
    UINT m_cubeIndexCount = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_cubeConstantBuffer;
    UINT8* m_cubeCBVDataBegin = nullptr;
    struct CubeConstants {
        DirectX::XMFLOAT4X4 model;
        DirectX::XMFLOAT4X4 view;
        DirectX::XMFLOAT4X4 proj;
    };
    void CreateCubeResources();
    void DrawCube(ID3D12GraphicsCommandList* commandList);
};
