#include "umgebung/core/EngineContext.hpp"
#include <spdlog/spdlog.h>

#if UMGEBUNG_HAS_GUI
#include "umgebung/graphics/RenderDevice.hpp"
#endif

namespace umgebung::core {

    bool EngineContext::Initialize(int argc, char** argv) {
        spdlog::info("Initializing Engine Context...");

        // 1. Profile system hardware (Common to all builds)
        m_hwCaps = SystemProfiler::ProfileSystem();

        if (m_hwCaps.cudaDeviceCount == 0) {
            spdlog::critical("No CUDA devices found. Aborting initialization.");
            return false;
        }

        // 2. Conditionally initialize the rendering device if GUI feature is active
#if UMGEBUNG_HAS_GUI
        spdlog::info("GUI mode enabled. Initializing NVRHI Render Device and window swapchain...");
        m_renderDevice = std::make_unique<graphics::RenderDevice>();
        if (!m_renderDevice->Initialize(m_hwCaps)) {
            spdlog::error("Failed to initialize RenderDevice.");
            return false;
        }
#else
        spdlog::info("Headless HPC mode active. Skipping NVRHI and window initialization.");
#endif

        m_isRunning = true;
        return true;
    }

    void EngineContext::Run() {
        spdlog::info("Entering main simulation loop...");

        while (m_isRunning) {
            // Main loop logic:
            // 1. Step CUDA physics solvers
            // 2. If GUI: render frame via NVRHI / ImGui
            // 3. If Headless: log stats or write checkpoint to disk

            // For now, break immediately for skeleton testing
            break;
        }
    }

    void EngineContext::Shutdown() {
        spdlog::info("Shutting down Engine Context...");
#if UMGEBUNG_HAS_GUI
        m_renderDevice.reset();
#endif
        m_isRunning = false;
    }

} // namespace umgebung::core