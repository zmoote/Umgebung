/**
 * @file Application.cpp
 * @brief Implementation of the `umgebung::core::Application` lifecycle orchestrator.
 *
 * @details
 * Implements CLI parsing (`cxxopts`), structured logging setup (`spdlog`),
 * startup hardware discovery and adaptation (`SystemProfiler`), active CUDA
 * device binding (`cudaSetDevice`), conditional desktop `Window` + `RenderDevice`
 * (Vulkan 1.2 + NVRHI + Dear ImGui) startup (compiled strictly when
 * `UMGEBUNG_HAS_GUI == 1`), and main-loop orchestration.
 */

#include "umgebung/core/Application.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
    #include "umgebung/graphics/RenderDevice.hpp"
    #include "umgebung/graphics/Window.hpp"
    #include <imgui.h>
#endif

namespace umgebung::core {

    // Constructor and destructor are defined in this `.cpp` where `Window` and
    // `RenderDevice` are complete types (in GUI builds) so `std::unique_ptr`
    // destructors compile without leaking graphics headers into `Application.hpp`.
    Application::Application() = default;

    Application::~Application() {
        Shutdown();
    }

    ApplicationInitResult Application::ParseCommandLine(int argc, char** argv) {
        try {
            cxxopts::Options cli(
                "Umgebung",
                "GPU-accelerated multi-scale reality simulation engine and computational physics testbed"
            );

            cli.add_options()
                ("h,help", "Display command-line usage and exit")
                ("v,version", "Display engine version and build configuration, then exit")
                ("headless", "Run in headless compute mode without opening a desktop window")
                ("profile-only", "Detect hardware, compute the startup adaptation plan, and exit")
                ("json-profile", "Print the hardware inventory and adaptation plan as JSON to stdout")
                ("d,device", "Explicitly select a CUDA GPU device ordinal (0..N-1)", cxxopts::value<int>())
                ("width", "Initial desktop window width in pixels (GUI builds)", cxxopts::value<std::uint32_t>()->default_value("1280"))
                ("height", "Initial desktop window height in pixels (GUI builds)", cxxopts::value<std::uint32_t>()->default_value("720"))
                ("x11", "Force X11/XWayland GLFW backend on Linux instead of native Wayland (GUI builds)")
                ("log-level", "Logging verbosity (trace, debug, info, warn, error, off)", cxxopts::value<std::string>()->default_value("info"));

            const cxxopts::ParseResult result = cli.parse(argc, argv);

            if (result.count("help") > 0U) {
                m_options.showHelp = true;
                std::cout << cli.help() << std::endl;
                return ApplicationInitResult::EarlyExitSuccess;
            }

            if (result.count("version") > 0U) {
                m_options.showVersion = true;
                std::cout << "Umgebung Simulation Engine v" << kEngineVersion
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
                          << " [Desktop GUI Build]"
#else
                          << " [Headless HPC Build]"
#endif
                          << std::endl;
                return ApplicationInitResult::EarlyExitSuccess;
            }

            m_options.forceHeadless = (result.count("headless") > 0U);
            m_options.profileOnly = (result.count("profile-only") > 0U);
            m_options.dumpHardwareJson = (result.count("json-profile") > 0U);
            m_options.forceX11 = (result.count("x11") > 0U);
            m_options.windowWidth = result["width"].as<std::uint32_t>();
            m_options.windowHeight = result["height"].as<std::uint32_t>();
            m_options.logLevel = result["log-level"].as<std::string>();

            if (result.count("device") > 0U) {
                m_options.preferredGpuDevice = result["device"].as<int>();
            }

            if (m_options.windowWidth == 0U || m_options.windowHeight == 0U) {
                spdlog::error(
                    "Invalid window dimensions requested ({}x{}). Width and height must be > 0.",
                    m_options.windowWidth,
                    m_options.windowHeight
                );
                return ApplicationInitResult::Failure;
            }

            return ApplicationInitResult::Ready;
        } catch (const cxxopts::exceptions::exception& ex) {
            std::cerr << "Command-line argument error: " << ex.what() << "\n"
                      << "Run with --help for usage information." << std::endl;
            return ApplicationInitResult::Failure;
        }
    }

    void Application::ConfigureLogging() const {
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        if (m_options.logLevel == "trace") {
            spdlog::set_level(spdlog::level::trace);
        } else if (m_options.logLevel == "debug") {
            spdlog::set_level(spdlog::level::debug);
        } else if (m_options.logLevel == "warn") {
            spdlog::set_level(spdlog::level::warn);
        } else if (m_options.logLevel == "error") {
            spdlog::set_level(spdlog::level::err);
        } else if (m_options.logLevel == "off") {
            spdlog::set_level(spdlog::level::off);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
    }

    ApplicationInitResult Application::Initialize(int argc, char** argv) {
        // ---------------------------------------------------------------------
        // Stage 1: Parse Command-Line Options & Configure Structured Logging
        // ---------------------------------------------------------------------
        const ApplicationInitResult cliStatus = ParseCommandLine(argc, argv);
        if (cliStatus != ApplicationInitResult::Ready) {
            return cliStatus;
        }

        ConfigureLogging();

        spdlog::info("Starting Umgebung Reality Simulation Engine v{}", kEngineVersion);

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        m_guiActive = (!m_options.forceHeadless && !m_options.profileOnly);
#else
        m_guiActive = false;
#endif

        // ---------------------------------------------------------------------
        // Stage 2: Startup Hardware Discovery & Runtime Adaptation
        // ---------------------------------------------------------------------
        m_hardwareCaps = SystemProfiler::ProfileSystem();

        HardwareAdaptationConfig adaptConfig{};
        adaptConfig.guiActive = m_guiActive;
        adaptConfig.preferredDeviceIndex = m_options.preferredGpuDevice;

        m_adaptationPlan = SystemProfiler::BuildAdaptationPlan(m_hardwareCaps, adaptConfig);
        SystemProfiler::LogSummary(m_hardwareCaps, m_adaptationPlan);

        if (m_options.dumpHardwareJson) {
            std::cout << SystemProfiler::ToJson(m_hardwareCaps, m_adaptationPlan).dump(2) << std::endl;
        }

        if (!m_adaptationPlan.computeReady || m_adaptationPlan.primaryGpuIndex < 0) {
            spdlog::critical(
                "No usable NVIDIA CUDA GPU detected ({}). Umgebung requires CUDA acceleration.",
                m_hardwareCaps.cudaErrorMessage
            );
            return ApplicationInitResult::Failure;
        }

        const cudaError_t setDevStatus = cudaSetDevice(m_adaptationPlan.primaryGpuIndex);
        if (setDevStatus != cudaSuccess) {
            spdlog::critical(
                "Failed to bind CUDA runtime to primary GPU {}: {}",
                m_adaptationPlan.primaryGpuIndex,
                cudaGetErrorString(setDevStatus)
            );
            return ApplicationInitResult::Failure;
        }

        spdlog::info("Bound CUDA runtime to adapted primary GPU [{}].", m_adaptationPlan.primaryGpuIndex);

        // ---------------------------------------------------------------------
        // Stage 3: Desktop Window & Vulkan/NVRHI RenderDevice (GUI Builds Only)
        // ---------------------------------------------------------------------
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        if (m_guiActive) {
            std::string windowTitle = std::string("Umgebung v") + kEngineVersion;
            for (const auto& gpu : m_hardwareCaps.gpus) {
                if (gpu.deviceIndex == m_adaptationPlan.primaryGpuIndex) {
                    windowTitle += " | " + gpu.name + " (SM " +
                                   std::to_string(gpu.computeCapabilityMajor) + "." +
                                   std::to_string(gpu.computeCapabilityMinor) + ")";
                    break;
                }
            }

            graphics::WindowConfig winConfig{};
            winConfig.title = std::move(windowTitle);
            winConfig.width = m_options.windowWidth;
            winConfig.height = m_options.windowHeight;
            winConfig.resizable = true;
            winConfig.preferX11OnLinux = m_options.forceX11;

            m_window = std::make_unique<graphics::Window>();
            if (!m_window->Initialize(winConfig)) {
                spdlog::error("Desktop Window initialization failed. Aborting startup.");
                m_window.reset();
                return ApplicationInitResult::Failure;
            }

            m_renderDevice = std::make_unique<graphics::RenderDevice>();
            if (!m_renderDevice->Initialize(*m_window, m_hardwareCaps, m_adaptationPlan)) {
                spdlog::error("RenderDevice initialization failed. Aborting startup.");
                m_renderDevice.reset();
                m_window->Shutdown();
                m_window.reset();
                return ApplicationInitResult::Failure;
            }
        }
#endif

        m_initialized = true;
        return ApplicationInitResult::Ready;
    }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
    void Application::RenderDesktopUi() {
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

        // 1. Fullscreen transparent dockspace allowing panels to dock cleanly around the viewport.
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        // 2. Top Main Menu Bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit", "Alt+F4")) {
                    m_window->RequestClose();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Hardware & Adaptation", nullptr, &m_showHardwarePanel);
                ImGui::MenuItem("Multiverse Hierarchy", nullptr, &m_showHierarchyPanel);
                ImGui::EndMenu();
            }

            // Right-aligned live FPS / frame-time readout in the menu bar.
            const ImGuiIO& io = ImGui::GetIO();
            const float frameMs = (io.Framerate > 0.0f) ? (1000.0f / io.Framerate) : 0.0f;
            ImGui::Separator();
            ImGui::Text(
                "  %.1f FPS (%.2f ms) | Swapchain %ux%u | %s",
                io.Framerate,
                frameMs,
                m_renderDevice->GetSwapchainWidth(),
                m_renderDevice->GetSwapchainHeight(),
                SystemProfiler::HardwareTierToString(m_adaptationPlan.tier)
            );
            ImGui::EndMainMenuBar();
        }

        // 3. Hardware & Adaptation Telemetry Panel
        if (m_showHardwarePanel) {
            ImGui::SetNextWindowSize(ImVec2(500.0f, 540.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Hardware & Adaptation", &m_showHardwarePanel)) {
                ImGui::Text("Engine Version : v%s", kEngineVersion);
                ImGui::Text("Execution Mode : %s", m_adaptationPlan.executionModeLabel.c_str());
                ImGui::Text("Hardware Tier  : %s", SystemProfiler::HardwareTierToString(m_adaptationPlan.tier));
                ImGui::Separator();

                ImGui::Text("Host OS        : %s", m_hardwareCaps.osName.c_str());
                ImGui::Text("CPU            : %s", m_hardwareCaps.cpuModelName.c_str());
                ImGui::Text(
                    "CPU Threads    : %u total (%u adapted worker threads)",
                    m_hardwareCaps.cpuLogicalCores,
                    m_adaptationPlan.recommendedHostWorkerThreads
                );

                const double totalRamGiB = static_cast<double>(m_hardwareCaps.totalSystemRAMBytes) / kBytesPerGiB;
                const double availRamGiB = static_cast<double>(m_hardwareCaps.availableSystemRAMBytes) / kBytesPerGiB;
                const double usedRamGiB = (totalRamGiB > availRamGiB) ? (totalRamGiB - availRamGiB) : 0.0;
                const float ramFrac = (totalRamGiB > 0.0) ? static_cast<float>(usedRamGiB / totalRamGiB) : 0.0f;

                ImGui::Text("Host RAM       : %.2f / %.2f GiB used (%.2f GiB available)", usedRamGiB, totalRamGiB, availRamGiB);
                ImGui::ProgressBar(ramFrac, ImVec2(-1.0f, 0.0f));
                ImGui::Separator();

                ImGui::Text(
                    "CUDA Toolchain : Driver %s | Runtime %s",
                    SystemProfiler::FormatCudaVersion(m_hardwareCaps.cudaDriverVersion).c_str(),
                    SystemProfiler::FormatCudaVersion(m_hardwareCaps.cudaRuntimeVersion).c_str()
                );
                ImGui::Text(
                    "Vulkan Device  : %s (%s)",
                    m_renderDevice->GetSelectedVulkanGpuName().c_str(),
                    m_renderDevice->IsMatchedToCudaGpu() ? "UUID Matched to CUDA" : "Fallback Device"
                );

                for (const auto& gpu : m_hardwareCaps.gpus) {
                    const bool isPrimary = (gpu.deviceIndex == m_adaptationPlan.primaryGpuIndex);
                    const double totalVramGiB = static_cast<double>(gpu.totalVRAMBytes) / kBytesPerGiB;
                    const double freeVramGiB = static_cast<double>(gpu.freeVRAMBytes) / kBytesPerGiB;
                    const double usedVramGiB = (totalVramGiB > freeVramGiB) ? (totalVramGiB - freeVramGiB) : 0.0;
                    const float vramFrac = (totalVramGiB > 0.0) ? static_cast<float>(usedVramGiB / totalVramGiB) : 0.0f;

                    ImGui::Spacing();
                    if (ImGui::CollapsingHeader(
                            (std::string("GPU [") + std::to_string(gpu.deviceIndex) + "] " + gpu.name +
                             (isPrimary ? " [PRIMARY]" : "")).c_str(),
                            ImGuiTreeNodeFlags_DefaultOpen
                        )) {
                        ImGui::Text(
                            "  Architecture : %s (SM %d.%d) | %d SMs",
                            gpu.architectureName.c_str(),
                            gpu.computeCapabilityMajor,
                            gpu.computeCapabilityMinor,
                            gpu.multiProcessorCount
                        );
                        ImGui::Text("  VRAM         : %.2f / %.2f GiB used (%.2f GiB free)", usedVramGiB, totalVramGiB, freeVramGiB);
                        ImGui::ProgressBar(vramFrac, ImVec2(-1.0f, 0.0f));
                        ImGui::Text("  Hardware UUID: %s", SystemProfiler::FormatUuid(gpu.uuid).c_str());
                    }
                }

                ImGui::Separator();
                ImGui::Text("Adapted Simulation Budgets:");
                ImGui::BulletText(
                    "Sim VRAM Budget : %.2f GiB (%.2f GiB GUI headroom reserved)",
                    static_cast<double>(m_adaptationPlan.simulationVRAMBudgetBytes) / kBytesPerGiB,
                    static_cast<double>(m_adaptationPlan.reservedVRAMHeadroomBytes) / kBytesPerGiB
                );
                ImGui::BulletText(
                    "CUDA Defaults   : blockSize=%d, streams=%d, FP64=%s",
                    m_adaptationPlan.recommendedCudaBlockSize,
                    m_adaptationPlan.recommendedCudaStreams,
                    m_adaptationPlan.useDoublePrecisionPhysics ? "ON" : "OFF"
                );
                ImGui::BulletText(
                    "Max FP64 Bodies : ~%llu",
                    static_cast<unsigned long long>(m_adaptationPlan.estimatedMaxFP64Bodies)
                );

                if (!m_adaptationPlan.adaptationNotes.empty()) {
                    ImGui::Separator();
                    ImGui::Text("Adaptation Notes:");
                    for (const auto& note : m_adaptationPlan.adaptationNotes) {
                        ImGui::TextWrapped("- %s", note.c_str());
                    }
                }
            }
            ImGui::End();
        }

        // 4. Multiverse Hierarchy Inspector (Scaffold for `.umgebung` scene loading)
        if (m_showHierarchyPanel) {
            ImGui::SetNextWindowSize(ImVec2(320.0f, 400.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Multiverse Hierarchy", &m_showHierarchyPanel)) {
                if (ImGui::TreeNodeEx("Multiverse (Source)", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (ImGui::TreeNodeEx("Universe 0 [Mainstream Physics Profile]", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::BulletText("No .umgebung scene loaded yet");
                        ImGui::TreePop();
                    }
                    if (ImGui::TreeNodeEx("Universe 1 [Unified Research Profile]")) {
                        ImGui::BulletText("Ready for Thinkers model integration");
                        ImGui::TreePop();
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::End();
        }
    }
#endif

    int Application::Run() {
        if (!m_initialized) {
            spdlog::error("Application::Run() invoked before successful Initialize().");
            return 1;
        }

        if (m_options.profileOnly) {
            spdlog::info("Profile-only mode complete; exiting cleanly.");
            return 0;
        }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        if (m_guiActive && m_window != nullptr && m_renderDevice != nullptr) {
            spdlog::info("Entering interactive Vulkan/NVRHI/ImGui render loop (close the window to exit)...");

            // Main interactive render loop paced smoothly by Vulkan FIFO VSync.
            // Submitting and presenting frame #1 also commits the first buffer on
            // native Wayland so the compositor immediately maps the window.
            while (!m_window->ShouldClose()) {
                if (m_window->IsMinimized()) {
                    m_window->WaitEvents();
                    continue;
                }

                m_window->PollEvents();

                if (m_renderDevice->BeginFrame()) {
                    RenderDesktopUi();
                    m_renderDevice->EndFrameAndPresent();
                }
            }

            spdlog::info("Window close requested; exiting interactive desktop loop.");
            return 0;
        }
#endif

        spdlog::info("Running in Headless Compute mode (ready for .umgebung scene execution).");
        return 0;
    }

    void Application::Shutdown() noexcept {
        if (!m_initialized) {
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
            m_renderDevice.reset();
            m_window.reset();
#endif
            return;
        }

        spdlog::info("Shutting down Umgebung Application...");

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        // Strict teardown order: destroy RenderDevice (swapchain + VkSurfaceKHR +
        // ImGui) BEFORE destroying the native GLFW Window.
        if (m_renderDevice != nullptr) {
            m_renderDevice->Shutdown();
            m_renderDevice.reset();
        }
        if (m_window != nullptr) {
            m_window->Shutdown();
            m_window.reset();
        }
#endif

        m_guiActive = false;
        m_initialized = false;
        spdlog::info("Umgebung Application shutdown complete.");
    }

} // namespace umgebung::core
