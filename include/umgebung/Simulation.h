#pragma once

#include "../walbourn/DeviceResources.h"
#include "../walbourn/StepTimer.h"

#include <memory>

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

        void Clear();

        void CreateDeviceDependentResources();
        void CreateWindowSizeDependentResources();

        // Device resources.
        std::unique_ptr<DX::DeviceResources>        m_deviceResources;

        // Rendering loop timer.
        DX::StepTimer                               m_timer;

        // If using the DirectX Tool Kit for DX12, uncomment this line:
        std::unique_ptr<DirectX::GraphicsMemory> m_graphicsMemory;
    };
}