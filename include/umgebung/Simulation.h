#pragma once

#include "../walbourn/DeviceResources.h"
#include "../walbourn/StepTimer.h"

#include <memory>

// ImGui includes
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

namespace Umgebung {
    class Simulation final : public DX::IDeviceNotify
    {
    public:

        Simulation() noexcept(false);
        ~Simulation();

        Simulation(Simulation&&) = default;
        Simulation& operator= (Simulation&&) = default;

        Simulation(Simulation const&) = delete;
        Simulation& operator= (Simulation const&) = delete;

        // Initialization and management
        void Initialize(HWND window, int width, int height);

        // Basic simulation loop
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
        void GetDefaultSize(int& width, int& height) const noexcept;

    private:

        void Update(DX::StepTimer const& timer);
        void Render();
        void RenderImGui();

        void Clear();

        void CreateDeviceDependentResources();
        void CreateWindowSizeDependentResources();

        // ImGui setup and cleanup
        void InitializeImGui(HWND window);
        void CleanupImGui();

        // Device resources.
        std::unique_ptr<DX::DeviceResources>        m_deviceResources;

        // Rendering loop timer.
        DX::StepTimer                               m_timer;

        // DirectX Tool Kit for DX12
        std::unique_ptr<DirectX::GraphicsMemory>    m_graphicsMemory;

        // ImGui resources
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_imguiSrvHeap;
        bool m_imguiInitialized;
    };
}