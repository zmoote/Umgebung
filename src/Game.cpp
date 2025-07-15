//
// Game.cpp
//

#include "../include/pch.h"
#include "../include/Game.h"
#include "../include/Camera.h"
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")

extern void ExitGame() noexcept;

using namespace DirectX;

using Microsoft::WRL::ComPtr;

Game::Game() noexcept(false)
{
    m_deviceResources = std::make_unique<DX::DeviceResources>();
    // TODO: Provide parameters for swapchain format, depth/stencil format, and backbuffer count.
    //   Add DX::DeviceResources::c_AllowTearing to opt-in to variable rate displays.
    //   Add DX::DeviceResources::c_EnableHDR for HDR10 display.
    //   Add DX::DeviceResources::c_ReverseDepth to optimize depth buffer clears for 0 instead of 1.
    m_deviceResources->RegisterDeviceNotify(this);
}

Game::~Game()
{
    if (m_deviceResources)
    {
        m_deviceResources->WaitForGpu();
    }
}

// Initialize the Direct3D resources required to run.
void Game::Initialize(HWND window, int width, int height)
{
    m_deviceResources->SetWindow(window, width, height);

    m_deviceResources->CreateDeviceResources();
    CreateDeviceDependentResources();

    m_deviceResources->CreateWindowSizeDependentResources();
    CreateWindowSizeDependentResources();

    // TODO: Change the timer settings if you want something other than the default variable timestep mode.
    // e.g. for 60 FPS fixed timestep update logic, call:
    /*
    m_timer.SetFixedTimeStep(true);
    m_timer.SetTargetElapsedSeconds(1.0 / 60);
    */
}

#pragma region Frame Update
// Executes the basic game loop.
void Game::Tick()
{
    m_timer.Tick([&]()
    {
        Update(m_timer);
    });

    Render();
}

// Updates the world.
void Game::Update(DX::StepTimer const& timer)
{
    PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

    float elapsedTime = float(timer.GetElapsedSeconds());

    // TODO: Add your game logic here.
    elapsedTime;

    PIXEndEvent();
}
#pragma endregion

#pragma region Frame Render
// Draws the scene.
void Game::Render()
{
    // Don't try to render anything before the first Update.
    if (m_timer.GetFrameCount() == 0)
    {
        return;
    }

    // Prepare the command list to render a new frame.
    m_deviceResources->Prepare();
    Clear();

    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render");

    DrawCube(commandList);

    PIXEndEvent(commandList);

    // Show the new frame.
    PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");
    m_deviceResources->Present();

    // If using the DirectX Tool Kit for DX12, uncomment this line:
    m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());

    PIXEndEvent(m_deviceResources->GetCommandQueue());
}

// Helper method to clear the back buffers.
void Game::Clear()
{
    auto commandList = m_deviceResources->GetCommandList();
    PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Clear");

    // Clear the views.
    const auto rtvDescriptor = m_deviceResources->GetRenderTargetView();
    const auto dsvDescriptor = m_deviceResources->GetDepthStencilView();

    commandList->OMSetRenderTargets(1, &rtvDescriptor, FALSE, &dsvDescriptor);
    commandList->ClearRenderTargetView(rtvDescriptor, Colors::CornflowerBlue, 0, nullptr);
    commandList->ClearDepthStencilView(dsvDescriptor, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // Set the viewport and scissor rect.
    const auto viewport = m_deviceResources->GetScreenViewport();
    const auto scissorRect = m_deviceResources->GetScissorRect();
    commandList->RSSetViewports(1, &viewport);
    commandList->RSSetScissorRects(1, &scissorRect);

    PIXEndEvent(commandList);
}
#pragma endregion

#pragma region Message Handlers
// Message handlers
void Game::OnActivated()
{
    // TODO: Game is becoming active window.
}

void Game::OnDeactivated()
{
    // TODO: Game is becoming background window.
}

void Game::OnSuspending()
{
    // TODO: Game is being power-suspended (or minimized).
}

void Game::OnResuming()
{
    m_timer.ResetElapsedTime();

    // TODO: Game is being power-resumed (or returning from minimize).
}

void Game::OnWindowMoved()
{
    const auto r = m_deviceResources->GetOutputSize();
    m_deviceResources->WindowSizeChanged(r.right, r.bottom);
}

void Game::OnDisplayChange()
{
    m_deviceResources->UpdateColorSpace();
}

void Game::OnWindowSizeChanged(int width, int height)
{
    if (!m_deviceResources->WindowSizeChanged(width, height))
        return;

    CreateWindowSizeDependentResources();

    // TODO: Game window is being resized.
}

// Properties
void Game::GetDefaultSize(int& width, int& height) const noexcept
{
    // TODO: Change to desired default window size (note minimum size is 320x200).
    width = 800;
    height = 600;
}
#pragma endregion

#pragma region Direct3D Resources
// These are the resources that depend on the device.
void Game::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Check Shader Model 6 support
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_0 };
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)))
        || (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_0))
    {
#ifdef _DEBUG
        OutputDebugStringA("ERROR: Shader Model 6.0 is not supported!\n");
#endif
        throw std::runtime_error("Shader Model 6.0 is not supported!");
    }

    // If using the DirectX Tool Kit for DX12, uncomment this line:
    m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

    CreateCubeResources();

    // TODO: Initialize device dependent objects here (independent of window size).
}

void Game::CreateCubeResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Vertex data for a colored cube
    struct Vertex { DirectX::XMFLOAT3 pos; DirectX::XMFLOAT3 color; };
    Vertex vertices[] = {
        // Front
        {{-1,-1,-1},{1,0,0}},{{-1,1,-1},{0,1,0}},{{1,1,-1},{0,0,1}},{{1,-1,-1},{1,1,0}},
        // Back
        {{-1,-1,1},{1,0,1}},{{-1,1,1},{0,1,1}},{{1,1,1},{1,1,1}},{{1,-1,1},{0,0,0}},
    };
    uint16_t indices[] = {
        0,1,2, 0,2,3, // Front
        4,6,5, 4,7,6, // Back
        4,5,1, 4,1,0, // Left
        3,2,6, 3,6,7, // Right
        1,5,6, 1,6,2, // Top
        4,0,3, 4,3,7  // Bottom
    };
    m_cubeIndexCount = _countof(indices);

    // Create vertex buffer
    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(vertices));
    device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &vbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_cubeVertexBuffer));
    void* pVertexData;
    m_cubeVertexBuffer->Map(0, nullptr, &pVertexData);
    memcpy(pVertexData, vertices, sizeof(vertices));
    m_cubeVertexBuffer->Unmap(0, nullptr);
    m_cubeVBV.BufferLocation = m_cubeVertexBuffer->GetGPUVirtualAddress();
    m_cubeVBV.StrideInBytes = sizeof(Vertex);
    m_cubeVBV.SizeInBytes = sizeof(vertices);

    // Create index buffer
    CD3DX12_RESOURCE_DESC ibDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(indices));
    device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &ibDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_cubeIndexBuffer));
    void* pIndexData;
    m_cubeIndexBuffer->Map(0, nullptr, &pIndexData);
    memcpy(pIndexData, indices, sizeof(indices));
    m_cubeIndexBuffer->Unmap(0, nullptr);
    m_cubeIBV.BufferLocation = m_cubeIndexBuffer->GetGPUVirtualAddress();
    m_cubeIBV.Format = DXGI_FORMAT_R16_UINT;
    m_cubeIBV.SizeInBytes = sizeof(indices);

    // Create constant buffer
    CD3DX12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer((sizeof(CubeConstants) + 255) & ~255);
    device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &cbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_cubeConstantBuffer));
    CD3DX12_RANGE readRange(0, 0);
    m_cubeConstantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_cubeCBVDataBegin));

    // Compile shaders
    ComPtr<ID3DBlob> vsBlob, psBlob;
    D3DCompileFromFile(L"assets/shaders/CubeVS.hlsl", nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsBlob, nullptr);
    D3DCompileFromFile(L"assets/shaders/CubePS.hlsl", nullptr, nullptr, "main", "ps_5_0", 0, 0, &psBlob, nullptr);

    // Input layout
    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
        {"COLOR",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}
    };

    // Root signature
    CD3DX12_ROOT_PARAMETER rootParams[1];
    rootParams[0].InitAsConstantBufferView(0);
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(1, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    ComPtr<ID3DBlob> sigBlob, errBlob;
    D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&m_cubeRootSignature));

    // Pipeline state
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
    psoDesc.pRootSignature = m_cubeRootSignature.Get();
    psoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
    psoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = m_deviceResources->GetBackBufferFormat();
    psoDesc.DSVFormat = m_deviceResources->GetDepthBufferFormat();
    psoDesc.SampleDesc.Count = 1;
    device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_cubePipelineState));
}

void Game::DrawCube(ID3D12GraphicsCommandList* commandList)
{
    // Update constant buffer
    CubeConstants cb;
    XMStoreFloat4x4(&cb.model, XMMatrixIdentity());
    XMStoreFloat4x4(&cb.view, m_camera.GetViewMatrix());
    RECT rc = m_deviceResources->GetOutputSize();
    float aspect = float(rc.right - rc.left) / float(rc.bottom - rc.top);
    XMStoreFloat4x4(&cb.proj, XMMatrixTranspose(m_camera.GetProjectionMatrix(aspect)));
    memcpy(m_cubeCBVDataBegin, &cb, sizeof(cb));

    // Set pipeline
    commandList->SetPipelineState(m_cubePipelineState.Get());
    commandList->SetGraphicsRootSignature(m_cubeRootSignature.Get());
    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList->IASetVertexBuffers(0, 1, &m_cubeVBV);
    commandList->IASetIndexBuffer(&m_cubeIBV);
    commandList->SetGraphicsRootConstantBufferView(0, m_cubeConstantBuffer->GetGPUVirtualAddress());
    commandList->DrawIndexedInstanced(m_cubeIndexCount, 1, 0, 0, 0);
}

// Allocate all memory resources that change on a window SizeChanged event.
void Game::CreateWindowSizeDependentResources()
{
    // TODO: Initialize windows-size dependent objects here.
}

void Game::OnDeviceLost()
{
    // TODO: Add Direct3D resource cleanup here.

    // If using the DirectX Tool Kit for DX12, uncomment this line:
    m_graphicsMemory.reset();
}

void Game::OnDeviceRestored()
{
    CreateDeviceDependentResources();

    CreateWindowSizeDependentResources();
}
#pragma endregion

Camera& Game::GetCamera() { return m_camera; }

void Game::OnCameraInput(float dx, float dy, bool mouse, float dt)
{
    if (mouse) {
        // Mouse look: dx = deltaX, dy = deltaY
        float sensitivity = 0.002f;
        m_camera.Rotate(dy * sensitivity, dx * sensitivity);
    } else {
        // Keyboard move: dx = strafe, dy = forward
        float speed = 5.0f * dt;
        DirectX::XMFLOAT3 move(dx * speed, 0, dy * speed);
        m_camera.Move(move);
    }
}
