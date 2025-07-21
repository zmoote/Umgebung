#define NOMINMAX
#include <windows.h>

#include "../include/umgebung/Camera.hpp"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <DirectXTK12/SpriteBatch.h>
#include <DirectXTK12/Model.h>
#include <DirectXTK12/CommonStates.h>
#include <DirectXTK12/GraphicsMemory.h>
#include <DirectXTK12/DescriptorHeap.h>
#include <DirectXTK12/ResourceUploadBatch.h>
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <imgui_impl_win32.h>
#include <windows.h>
#include <wrl.h>
#include <stdexcept>
#include <memory>
#include <directx/d3dx12.h>
#include <algorithm>
#undef max
#undef min

// Export D3D12 Agility SDK version and path
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 616; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12"; }

using namespace Microsoft::WRL;

// Main application class
class UmgebungApp {
public:
    // Constructor: Initialize with window handle and dimensions
    UmgebungApp(HWND hwnd, int width, int height) : hwnd_(hwnd), width_(width), height_(height) {
        InitializeD3D12();
        InitializeDirectXTK();
        InitializeImGui();
        camera_ = std::make_unique<Umgebung::Camera>(static_cast<float>(width), static_cast<float>(height));
    }

    // Destructor: Clean up resources
    ~UmgebungApp() {
        WaitForPreviousFrame(); // Ensure GPU is idle
        ImGui_ImplDX12_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
        CloseHandle(fenceEvent_);
    }

    // Update: Handle input and UI
    void Update(float deltaTime) {
        static POINT lastMousePos = { 0, 0 };
        POINT currentMousePos;
        GetCursorPos(&currentMousePos);
        ScreenToClient(hwnd_, &currentMousePos);
        // Clamp mouse position to window size
        int mouseX = static_cast<int>(currentMousePos.x);
        int mouseY = static_cast<int>(currentMousePos.y);
        mouseX = std::clamp(mouseX, 0, static_cast<int>(width_ - 1));
        mouseY = std::clamp(mouseY, 0, static_cast<int>(height_ - 1));
        currentMousePos.x = mouseX;
        currentMousePos.y = mouseY;
        bool mouseDown = GetAsyncKeyState(VK_RBUTTON) & 0x8000;
        float mouseDeltaX = static_cast<float>(currentMousePos.x - lastMousePos.x) * 0.005f;
        float mouseDeltaY = static_cast<float>(currentMousePos.y - lastMousePos.y) * 0.005f;
        lastMousePos = currentMousePos;

        camera_->Update(deltaTime, mouseDown, mouseDeltaX, mouseDeltaY);

        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(width_), static_cast<float>(height_));
        io.MousePos = ImVec2(static_cast<float>(currentMousePos.x), static_cast<float>(currentMousePos.y));

        ImGui_ImplDX12_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Camera Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        if (ImGui::Button("Switch to 2D")) camera_->SetViewMode(Umgebung::Camera::ViewMode::View2D);
        if (ImGui::Button("Switch to 3D")) camera_->SetViewMode(Umgebung::Camera::ViewMode::View3D);
        ImGui::Text("Mouse: (%.1f, %.1f)", io.MousePos.x, io.MousePos.y);
        ImGui::End();
        ImGui::Render();
    }

    // Render: Draw the scene
    void Render() {
        commandAllocator_->Reset();
        commandList_->Reset(commandAllocator_.Get(), nullptr);

        // Transition render target to render state
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = renderTargets_[frameIndex_].Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        commandList_->ResourceBarrier(1, &barrier);

        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        rtvHandle.ptr += frameIndex_ * rtvDescriptorSize_;
        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap_->GetCPUDescriptorHandleForHeapStart();

        const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
        commandList_->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        commandList_->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        commandList_->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

        // Set the descriptor heap for ImGui
        ID3D12DescriptorHeap* heaps[] = { srvHeap_.Get() };
        commandList_->SetDescriptorHeaps(_countof(heaps), heaps);

        // Render based on view mode
        if (camera_->GetViewMode() == Umgebung::Camera::ViewMode::View2D) {
            ID3D12DescriptorHeap* heaps[] = { spriteSrvHeap_->Heap() };
            commandList_->SetDescriptorHeaps(_countof(heaps), heaps);
            spriteBatch_->Begin(commandList_.Get());
            DirectX::XMFLOAT2 pos(static_cast<float>(width_) / 2 - 50, static_cast<float>(height_) / 2 - 50);
            DirectX::XMFLOAT2 origin(0, 0);
            DirectX::XMFLOAT2 scale(100.0f, 100.0f);
            RECT destRect;
            destRect.left = static_cast<LONG>(width_ / 2 - 50);
            destRect.top = static_cast<LONG>(height_ / 2 - 50);
            destRect.right = destRect.left + 100;
            destRect.bottom = destRect.top + 100;
            spriteBatch_->Draw(
                dummyTextureGpuHandle_,
                DirectX::XMUINT2(1, 1),
                DirectX::XMFLOAT2(static_cast<float>(width_) / 2 - 50, static_cast<float>(height_) / 2 - 50),
                nullptr,
                DirectX::Colors::Orange,
                0.0f,
                DirectX::XMFLOAT2(0, 0),
                100.0f
            );
            spriteBatch_->End();
        }
        else {
            // Use identity matrices for demonstration
            DirectX::XMMATRIX world = DirectX::XMMatrixIdentity();
            DirectX::XMMATRIX view = DirectX::XMMatrixIdentity();
            DirectX::XMMATRIX proj = DirectX::XMMatrixIdentity();
            DirectX::XMMATRIX viewProj = view * proj;
            model_->Draw(commandList_.Get());
        }

        // Render ImGui UI
        ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList_.Get());

        // Transition render target to present state
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        commandList_->ResourceBarrier(1, &barrier);

        commandList_->Close();
        ID3D12CommandList* commandLists[] = { commandList_.Get() };
        commandQueue_->ExecuteCommandLists(1, commandLists);

        swapChain_->Present(1, 0);
        WaitForPreviousFrame();
        frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
    }

    // Handle window resize
    void OnResize(int newWidth, int newHeight) {
        width_ = newWidth;
        height_ = newHeight;
        // If your camera has width/height members, update them here
        // camera_->SetViewportSize(static_cast<float>(width_), static_cast<float>(height_)); // Remove this if not available
    }

private:
    // Initialize DirectX 12 resources
    void InitializeD3D12() {
        ComPtr<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            throw std::runtime_error("Failed to create DXGI factory");
        }
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)))) {
            throw std::runtime_error("Failed to create D3D12 device");
        }

        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue_)))) {
            throw std::runtime_error("Failed to create command queue");
        }

        DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
        swapChainDesc.BufferCount = FrameCount;
        swapChainDesc.Width = width_;
        swapChainDesc.Height = height_;
        swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.SampleDesc.Count = 1;
        ComPtr<IDXGISwapChain1> swapChain1;
        if (FAILED(factory->CreateSwapChainForHwnd(commandQueue_.Get(), hwnd_, &swapChainDesc, nullptr, nullptr, &swapChain1))) {
            throw std::runtime_error("Failed to create swap chain");
        }
        swapChain1.As(&swapChain_);
        frameIndex_ = swapChain_->GetCurrentBackBufferIndex();

        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
        rtvHeapDesc.NumDescriptors = FrameCount;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        if (FAILED(device_->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap_)))) {
            throw std::runtime_error("Failed to create RTV heap");
        }
        rtvDescriptorSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
        srvHeapDesc.NumDescriptors = 1;
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(device_->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap_)))) {
            throw std::runtime_error("Failed to create SRV heap");
        }

        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
        dsvHeapDesc.NumDescriptors = 1;
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        if (FAILED(device_->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap_)))) {
            throw std::runtime_error("Failed to create DSV heap");
        }

        for (UINT i = 0; i < FrameCount; i++) {
            if (FAILED(swapChain_->GetBuffer(i, IID_PPV_ARGS(&renderTargets_[i])))) {
                throw std::runtime_error("Failed to get swap chain buffer");
            }
            device_->CreateRenderTargetView(renderTargets_[i].Get(), nullptr, { rtvHeap_->GetCPUDescriptorHandleForHeapStart().ptr + i * rtvDescriptorSize_ });
        }

        D3D12_RESOURCE_DESC depthDesc = {};
        depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        depthDesc.Width = width_;
        depthDesc.Height = height_;
        depthDesc.DepthOrArraySize = 1;
        depthDesc.MipLevels = 1;
        depthDesc.Format = DXGI_FORMAT_D32_FLOAT;
        depthDesc.SampleDesc.Count = 1;
        depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_CLEAR_VALUE depthClear = {};
        depthClear.Format = DXGI_FORMAT_D32_FLOAT;
        depthClear.DepthStencil.Depth = 1.0f;
        // Define the heap properties as a named variable
        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);

        // Use the variable in the CreateCommittedResource call
        if (FAILED(device_->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &depthDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthClear, IID_PPV_ARGS(&depthStencil_)))) {
            throw std::runtime_error("Failed to create depth stencil resource");
        }
        device_->CreateDepthStencilView(depthStencil_.Get(), nullptr, dsvHeap_->GetCPUDescriptorHandleForHeapStart());

        if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator_)))) {
            throw std::runtime_error("Failed to create command allocator");
        }
        if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator_.Get(), nullptr, IID_PPV_ARGS(&commandList_)))) {
            throw std::runtime_error("Failed to create command list");
        }
        commandList_->Close();

        if (FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) {
            throw std::runtime_error("Failed to create fence");
        }
        fenceValue_ = 1;
        fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }

    // Initialize DirectXTK resources
    void InitializeDirectXTK() {
        graphicsMemory_ = std::make_unique<DirectX::GraphicsMemory>(device_.Get());
        commonStates_ = std::make_unique<DirectX::CommonStates>(device_.Get());
        resourceUploadBatch_ = std::make_unique<DirectX::ResourceUploadBatch>(device_.Get());
        resourceUploadBatch_->Begin(D3D12_COMMAND_LIST_TYPE_DIRECT);
        DirectX::RenderTargetState renderTargetState(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_D32_FLOAT);
        DirectX::DX12::SpriteBatchPipelineStateDescription spriteBatchDesc(renderTargetState);
        D3D12_VIEWPORT* viewport = nullptr;
        spriteBatch_ = std::make_unique<DirectX::DX12::SpriteBatch>(
            device_.Get(),
            *resourceUploadBatch_,
            spriteBatchDesc,
            viewport
        );
        // Create a 1x1 white texture for rectangle rendering
        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width = 1;
        texDesc.Height = 1;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
        device_->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&dummyTexture_));
        // Upload white pixel data
        D3D12_SUBRESOURCE_DATA textureData = {};
        UINT8 whitePixel[4] = { 255, 255, 255, 255 };
        textureData.pData = whitePixel;
        textureData.RowPitch = 4;
        textureData.SlicePitch = 4;
        resourceUploadBatch_->Upload(dummyTexture_.Get(), 0, &textureData, 1);
        resourceUploadBatch_->End(commandQueue_.Get()).wait();
        // Create SRV heap for sprite textures
        spriteSrvHeap_ = std::make_unique<DirectX::DescriptorHeap>(
            device_.Get(),
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            1
        );
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(dummyTexture_.Get(), &srvDesc, spriteSrvHeap_->GetCpuHandle(0));
        dummyTextureGpuHandle_ = spriteSrvHeap_->GetGpuHandle(0);
        model_ = DirectX::Model::CreateFromSDKMESH(device_.Get(), L"assets/models/tiny.sdkmesh");
        if (!model_) {
            throw std::runtime_error("Failed to load tiny.sdkmesh");
        }
    }

    // Initialize ImGui
    void InitializeImGui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplWin32_Init(hwnd_);
        ImGui_ImplDX12_Init(device_.Get(), FrameCount, DXGI_FORMAT_R8G8B8A8_UNORM, srvHeap_.Get(),
            srvHeap_->GetCPUDescriptorHandleForHeapStart(), srvHeap_->GetGPUDescriptorHandleForHeapStart());
    }

    // Wait for the previous frame to complete
    void WaitForPreviousFrame() {
        const UINT64 fence = fenceValue_;
        commandQueue_->Signal(fence_.Get(), fence);
        fenceValue_++;
        if (fence_->GetCompletedValue() < fence) {
            fence_->SetEventOnCompletion(fence, fenceEvent_);
            WaitForSingleObject(fenceEvent_, INFINITE);
        }
    }

    // Member variables
    static const UINT FrameCount = 2;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> commandQueue_;
    ComPtr<IDXGISwapChain3> swapChain_;
    ComPtr<ID3D12DescriptorHeap> rtvHeap_;
    ComPtr<ID3D12DescriptorHeap> srvHeap_;
    ComPtr<ID3D12DescriptorHeap> dsvHeap_;
    ComPtr<ID3D12Resource> renderTargets_[FrameCount];
    ComPtr<ID3D12Resource> depthStencil_;
    ComPtr<ID3D12CommandAllocator> commandAllocator_;
    ComPtr<ID3D12GraphicsCommandList> commandList_;
    ComPtr<ID3D12Fence> fence_;
    UINT64 fenceValue_;
    HANDLE fenceEvent_;
    UINT frameIndex_;
    UINT rtvDescriptorSize_;
    std::unique_ptr<DirectX::GraphicsMemory> graphicsMemory_;
    std::unique_ptr<DirectX::CommonStates> commonStates_;
    std::unique_ptr<DirectX::SpriteBatch> spriteBatch_;
    std::unique_ptr<DirectX::Model> model_;
    std::unique_ptr<Umgebung::Camera> camera_;
    // Add resourceUploadBatch_ member variable:
    std::unique_ptr<DirectX::ResourceUploadBatch> resourceUploadBatch_;
    std::unique_ptr<DirectX::DescriptorHeap> spriteSrvHeap_;
    D3D12_GPU_DESCRIPTOR_HANDLE dummyTextureGpuHandle_{};
    ComPtr<ID3D12Resource> dummyTexture_;
    int width_, height_;
    HWND hwnd_; // Store the window handle
};

// Windows entry point
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    // Register window class
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_HREDRAW | CS_VREDRAW, [](HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) -> LRESULT {
        static UmgebungApp* app = nullptr;
        switch (msg) {
        case WM_SIZE:
            if (app) {
                int newWidth = LOWORD(lParam);
                int newHeight = HIWORD(lParam);
                app->OnResize(newWidth, newHeight);
            }
            break;
        case WM_CREATE:
            {
                CREATESTRUCT* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
                app = reinterpret_cast<UmgebungApp*>(cs->lpCreateParams);
            }
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }, 0, 0, hInstance, nullptr, nullptr, nullptr, nullptr, L"Umgebung", nullptr };
    RegisterClassEx(&wc);

    // Create window
    int width = 1280, height = 720;
    HWND hwnd = CreateWindow(L"Umgebung", L"Umgebung", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height, nullptr, nullptr, hInstance, nullptr);
    ShowWindow(hwnd, nCmdShow);

    // Initialize application
    UmgebungApp app(hwnd, width, height);

    // Main loop
    MSG msg = {};
    float deltaTime = 1.0f / 60.0f; // Fixed timestep for simplicity
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else {
            app.Update(deltaTime);
            app.Render();
        }
    }

    return static_cast<int>(msg.wParam);
}