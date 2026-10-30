#pragma once

#include "umgebung/core/SystemProfiler.hpp"
#include <memory>

#if UMGEBUNG_HAS_GUI
namespace umgebung::graphics {
    class RenderDevice; // Forward declaration
}
#endif

namespace umgebung::core {

    class EngineContext {
    public:
        EngineContext() = default;
        ~EngineContext() = default;

        bool Initialize(int argc, char** argv);
        void Run();
        void Shutdown();

        const HardwareCapabilities& GetHardwareCapabilities() const { return m_hwCaps; }

#if UMGEBUNG_HAS_GUI
        graphics::RenderDevice* GetRenderDevice() const { return m_renderDevice.get(); }
#endif

    private:
        HardwareCapabilities m_hwCaps{};
        bool m_isRunning{false};

        // Graphics and windowing resources are conditionally compiled
#if UMGEBUNG_HAS_GUI
        std::unique_ptr<graphics::RenderDevice> m_renderDevice;
#endif
    };

} // namespace umgebung::core