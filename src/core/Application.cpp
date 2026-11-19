/**
 * @file Application.cpp
 * @brief Implementation of `umgebung::core::Application` integrating startup
 *        hardware profiling/adaptation, `compute::CUDAContext`, `.umgebung`
 *        `scene::Multiverse` loading, Pillar 1's FP64 CUDA `NewtonianNBodySolver`,
 *        headless HPC batch execution, and the GUI-exclusive `Window` + `RenderDevice`.
 */

#include "umgebung/core/Application.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/SolverFactory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
    #include "umgebung/graphics/RenderDevice.hpp"
    #include "umgebung/graphics/Window.hpp"
    #include <imgui.h>
#endif

namespace umgebung::core {

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
                ("s,scene", "Path to a .umgebung scene file or directory of .umgebung files", cxxopts::value<std::string>())
                ("steps", "Override step count for headless batch simulation", cxxopts::value<std::uint64_t>())
                ("dt", "Override integration timestep dt (scene time units)", cxxopts::value<double>())
                ("o,output", "Write headless simulation summary & final state JSON to path", cxxopts::value<std::string>())
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
            if (result.count("scene") > 0U) {
                m_options.scenePath = std::filesystem::path(result["scene"].as<std::string>());
            }
            if (result.count("steps") > 0U) {
                m_options.overrideSteps = result["steps"].as<std::uint64_t>();
            }
            if (result.count("dt") > 0U) {
                const double dt = result["dt"].as<double>();
                if (!std::isfinite(dt) || dt <= 0.0) {
                    std::cerr << "Invalid --dt value (" << dt << "); must be finite and > 0." << std::endl;
                    return ApplicationInitResult::Failure;
                }
                m_options.overrideTimestep = dt;
            }
            if (result.count("output") > 0U) {
                m_options.outputPath = std::filesystem::path(result["output"].as<std::string>());
            }

            if (m_options.windowWidth == 0U || m_options.windowHeight == 0U) {
                std::cerr << "Invalid window dimensions (" << m_options.windowWidth << "x"
                          << m_options.windowHeight << "); width and height must be > 0." << std::endl;
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

    bool Application::LoadInitialMultiverse(const char* argv0) {
        m_multiverse.Clear();

        // Case 1: Explicit `--scene <path>` supplied on CLI (file or directory).
        if (m_options.scenePath.has_value()) {
            const std::filesystem::path& target = m_options.scenePath.value();
            std::error_code ec;
            if (std::filesystem::is_directory(target, ec)) {
                const std::size_t loaded = m_multiverse.LoadDirectory(target);
                if (loaded == 0U) {
                    spdlog::error("No valid .umgebung scenes found in directory '{}'.", target.string());
                    return false;
                }
                return true;
            }

            std::string errMsg;
            if (!m_multiverse.LoadUniverseFromFile(target, errMsg)) {
                spdlog::error("Failed loading requested scene '{}': {}", target.string(), errMsg);
                return false;
            }
            return true;
        }

        // Case 2: Auto-discover bundled `.umgebung` scenes from `<exe-dir>/scenes`
        // or `./scenes` without failing if neither exists yet.
        std::vector<std::filesystem::path> searchDirs;
        if (argv0 != nullptr && argv0[0] != '\0') {
            std::error_code ec;
            const std::filesystem::path exePath = std::filesystem::weakly_canonical(argv0, ec);
            if (!ec && exePath.has_parent_path()) {
                searchDirs.push_back(exePath.parent_path() / "scenes");
            }
        }
        searchDirs.push_back(std::filesystem::path("scenes"));

        for (const auto& dir : searchDirs) {
            const std::size_t count = m_multiverse.LoadDirectory(dir);
            if (count > 0U) {
                spdlog::info("Auto-discovered {} .umgebung universe scene(s) from '{}'.", count, dir.string());
                return true;
            }
        }

        spdlog::info("No default .umgebung scenes auto-discovered (pass --scene <file.umgebung> to load one).");
        return true;
    }

    bool Application::ActivateUniverseForSimulation(std::size_t universeIndex) {
        if (!m_multiverse.SetActiveUniverseIndex(universeIndex)) {
            return false;
        }

        const scene::UniverseScene* activeUniverse = m_multiverse.GetActiveUniverse();
        if (activeUniverse == nullptr || m_cudaContext == nullptr || !m_cudaContext->IsInitialized()) {
            return false;
        }

        m_activeTimestep = m_options.overrideTimestep.value_or(activeUniverse->physicsProfile.timestep);
        m_liveBodies = activeUniverse->bodies;

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        m_bodyTrails.clear();
        m_bodyTrails.resize(m_liveBodies.size());

        // Set a comfortable fixed initial camera half-extent from the scene's
        // starting body distribution so escaping stars move across a stationary
        // reference frame instead of shrinking the galaxy core to a dot.
        double initialMaxCoord = 1.0;
        for (const auto& b : m_liveBodies) {
            initialMaxCoord = std::max(
                initialMaxCoord,
                std::max(std::abs(b.position.x), std::abs(b.position.y))
            );
        }
        m_viewportHalfExtent = static_cast<float>(initialMaxCoord * 1.4);
#endif

        if (activeUniverse->bodies.empty()) {
            m_activeSolver.reset();
            return true;
        }

        m_activeSolver = models::CreateSolverForScene(*activeUniverse);
        if (m_activeSolver == nullptr || !m_activeSolver->Initialize(*m_cudaContext, *activeUniverse)) {
            spdlog::error("Failed to initialize physics solver for universe '{}'.", activeUniverse->name);
            m_activeSolver.reset();
            return false;
        }

        (void)m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics);
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        RecordLiveTrails();
#endif
        return true;
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

        // ---------------------------------------------------------------------
        // Stage 3: Initialize GPU Compute Context (`compute::CUDAContext`)
        // ---------------------------------------------------------------------
        m_cudaContext = std::make_unique<compute::CUDAContext>();
        if (!m_cudaContext->Initialize(m_hardwareCaps, m_adaptationPlan)) {
            spdlog::critical("Failed to initialize CUDAContext on adapted primary GPU.");
            m_cudaContext.reset();
            return ApplicationInitResult::Failure;
        }

        // In profile-only mode, skip scene loading and GUI startup cleanly.
        if (m_options.profileOnly) {
            m_initialized = true;
            return ApplicationInitResult::Ready;
        }

        // ---------------------------------------------------------------------
        // Stage 4: Load Multiverse `.umgebung` Scenes & Initialize Active Solver
        // ---------------------------------------------------------------------
        const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
        if (!LoadInitialMultiverse(argv0)) {
            Shutdown();
            return ApplicationInitResult::Failure;
        }

        if (!m_multiverse.IsEmpty()) {
            if (!ActivateUniverseForSimulation(0U)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }
        }

        // ---------------------------------------------------------------------
        // Stage 5: Desktop Window & Vulkan/NVRHI RenderDevice (GUI Builds Only)
        // ---------------------------------------------------------------------
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        if (m_guiActive) {
            const auto& primaryGpu = m_cudaContext->GetPrimaryGpuInfo();
            std::string windowTitle =
                std::string("Umgebung v") + kEngineVersion + " | " + primaryGpu.name +
                " (SM " + std::to_string(primaryGpu.computeCapabilityMajor) + "." +
                std::to_string(primaryGpu.computeCapabilityMinor) + ")";

            graphics::WindowConfig winConfig{};
            winConfig.title = std::move(windowTitle);
            winConfig.width = m_options.windowWidth;
            winConfig.height = m_options.windowHeight;
            winConfig.resizable = true;
            winConfig.preferX11OnLinux = m_options.forceX11;

            m_window = std::make_unique<graphics::Window>();
            if (!m_window->Initialize(winConfig)) {
                spdlog::error("Desktop Window initialization failed. Aborting startup.");
                Shutdown();
                return ApplicationInitResult::Failure;
            }

            m_renderDevice = std::make_unique<graphics::RenderDevice>();
            if (!m_renderDevice->Initialize(*m_window, m_hardwareCaps, m_adaptationPlan)) {
                spdlog::error("RenderDevice initialization failed. Aborting startup.");
                Shutdown();
                return ApplicationInitResult::Failure;
            }
        }
#endif

        m_initialized = true;
        return ApplicationInitResult::Ready;
    }

    int Application::RunHeadlessSimulation() {
        const scene::UniverseScene* activeUniverse = m_multiverse.GetActiveUniverse();
        if (activeUniverse == nullptr || m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
            spdlog::info("Running in Headless Compute mode with no active scene loaded.");
            return 0;
        }

        const std::uint64_t stepsToRun =
            m_options.overrideSteps.value_or(activeUniverse->physicsProfile.defaultSteps);
        const double dt = m_activeTimestep;

        const models::SolverDiagnostics initialDiag = m_latestDiagnostics;

        spdlog::info(
            "Headless Batch Simulation: '{}' ({} bodies, solver={}, profile={}, integrator={}, steps={}, dt={:.6g})",
            activeUniverse->name,
            m_activeSolver->GetBodyCount(),
            m_activeSolver->GetSolverName(),
            scene::PhysicsProfileTypeToString(activeUniverse->physicsProfile.type),
            scene::IntegratorTypeToString(activeUniverse->physicsProfile.integrator),
            stepsToRun,
            dt
        );
        spdlog::info(
            "  Initial Invariants -> E0={:.12g} (T={:.12g}, U={:.12g}) | |P0|={:.3e} | |L0|={:.12g}",
            initialDiag.totalEnergy,
            initialDiag.kineticEnergy,
            initialDiag.potentialEnergy,
            initialDiag.totalLinearMomentum.Norm(),
            initialDiag.totalAngularMomentum.Norm()
        );

        const auto tStart = std::chrono::steady_clock::now();
        if (!m_activeSolver->StepMultiple(stepsToRun, dt)) {
            spdlog::error("Headless GPU simulation stepping failed.");
            return 1;
        }
        const auto tEnd = std::chrono::steady_clock::now();
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        if (!m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics) ||
            !m_activeSolver->SyncToHost(m_liveBodies)) {
            spdlog::error("Failed retrieving final simulation diagnostics from GPU.");
            return 1;
        }

        const double usPerStep = (stepsToRun > 0ULL) ? ((elapsedMs * 1000.0) / static_cast<double>(stepsToRun)) : 0.0;
        spdlog::info(
            "  Completed {} steps in {:.3f} ms ({:.2f} us/step) | t={:.6f}",
            m_latestDiagnostics.stepCount,
            elapsedMs,
            usPerStep,
            m_latestDiagnostics.simulationTime
        );
        spdlog::info(
            "  Final Invariants   -> E ={:.12g} (rel drift={:.3e}) | |P|={:.3e} | |L|={:.12g}",
            m_latestDiagnostics.totalEnergy,
            m_latestDiagnostics.relativeEnergyDrift,
            m_latestDiagnostics.totalLinearMomentum.Norm(),
            m_latestDiagnostics.totalAngularMomentum.Norm()
        );
        if (m_latestDiagnostics.hasUnifiedTelemetry) {
            spdlog::info(
                "  Unified Telemetry  -> <P/P0>={:.8f} | <c'/c>={:.8f} | <dtau/dt>={:.6f} | <tau>={:.6f} | Phryll={:.4g}",
                m_latestDiagnostics.meanVacuumPressureRatio,
                m_latestDiagnostics.meanLocalLightSpeedRatio,
                m_latestDiagnostics.meanLocalTimeRate,
                m_latestDiagnostics.meanLocalProperTime,
                m_latestDiagnostics.totalPhryllFlux
            );
        }

        // Optionally export structured JSON telemetry & final state for HPC post-processing.
        if (m_options.outputPath.has_value()) {
            nlohmann::json bodiesJson = nlohmann::json::array();
            for (const auto& b : m_liveBodies) {
                bodiesJson.push_back({
                    {"id", b.id},
                    {"name", b.name},
                    {"mass", b.mass},
                    {"position", {b.position.x, b.position.y, b.position.z}},
                    {"velocity", {b.velocity.x, b.velocity.y, b.velocity.z}}
                });
            }

            nlohmann::json report{
                {"universe_id", activeUniverse->universeId},
                {"universe_name", activeUniverse->name},
                {"solver_name", m_activeSolver->GetSolverName()},
                {"physics_profile", scene::PhysicsProfileTypeToString(activeUniverse->physicsProfile.type)},
                {"integrator", scene::IntegratorTypeToString(activeUniverse->physicsProfile.integrator)},
                {"steps", m_latestDiagnostics.stepCount},
                {"dt", dt},
                {"simulation_time", m_latestDiagnostics.simulationTime},
                {"wall_time_ms", elapsedMs},
                {"initial_energy", initialDiag.totalEnergy},
                {"final_energy", m_latestDiagnostics.totalEnergy},
                {"relative_energy_drift", m_latestDiagnostics.relativeEnergyDrift},
                {"final_linear_momentum", {
                    m_latestDiagnostics.totalLinearMomentum.x,
                    m_latestDiagnostics.totalLinearMomentum.y,
                    m_latestDiagnostics.totalLinearMomentum.z
                }},
                {"final_angular_momentum", {
                    m_latestDiagnostics.totalAngularMomentum.x,
                    m_latestDiagnostics.totalAngularMomentum.y,
                    m_latestDiagnostics.totalAngularMomentum.z
                }},
                {"unified_telemetry", {
                    {"active", m_latestDiagnostics.hasUnifiedTelemetry},
                    {"mean_vacuum_pressure_ratio", m_latestDiagnostics.meanVacuumPressureRatio},
                    {"mean_local_light_speed_ratio", m_latestDiagnostics.meanLocalLightSpeedRatio},
                    {"mean_local_time_rate", m_latestDiagnostics.meanLocalTimeRate},
                    {"mean_local_proper_time", m_latestDiagnostics.meanLocalProperTime},
                    {"mean_holographic_phi", m_latestDiagnostics.meanHolographicPhi},
                    {"total_phryll_flux", m_latestDiagnostics.totalPhryllFlux}
                }},
                {"bodies", std::move(bodiesJson)},
                {"hardware", SystemProfiler::ToJson(m_hardwareCaps, m_adaptationPlan)}
            };

            std::ofstream outFile(m_options.outputPath.value());
            if (!outFile.is_open()) {
                spdlog::error("Failed opening --output path '{}'.", m_options.outputPath->string());
                return 1;
            }
            outFile << report.dump(2) << '\n';
            spdlog::info("Wrote headless simulation telemetry JSON to '{}'.", m_options.outputPath->string());
        }

        return 0;
    }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
    void Application::RecordLiveTrails() {
        if (m_bodyTrails.size() != m_liveBodies.size()) {
            m_bodyTrails.assign(m_liveBodies.size(), {});
        }
        for (std::size_t i = 0U; i < m_liveBodies.size(); ++i) {
            auto& trail = m_bodyTrails[i];
            trail.push_back(m_liveBodies[i].position);
            if (trail.size() > kMaxTrailPoints) {
                trail.pop_front();
            }
        }
    }

    void Application::RenderBodyHierarchyNode(
        const std::vector<scene::BodyEntity>& bodies,
        const std::string& parentId
    ) {
        for (std::size_t i = 0U; i < bodies.size(); ++i) {
            const auto& body = bodies[i];
            if (body.parentId != parentId) {
                continue;
            }

            // Check whether this entity has any child bodies in the universe.
            bool hasChildren = false;
            for (const auto& candidate : bodies) {
                if (candidate.parentId == body.id) {
                    hasChildren = true;
                    break;
                }
            }

            const ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_DefaultOpen |
                (hasChildren ? ImGuiTreeNodeFlags_None : ImGuiTreeNodeFlags_Leaf);

            const std::string label =
                body.name + " [" + body.category + "]##" + body.id;

            if (ImGui::TreeNodeEx(label.c_str(), flags)) {
                ImGui::TextDisabled(
                    "m=%.6g | r=(%.3f, %.3f, %.3f)",
                    body.mass,
                    body.position.x,
                    body.position.y,
                    body.position.z
                );
                if (hasChildren) {
                    RenderBodyHierarchyNode(bodies, body.id);
                }
                ImGui::TreePop();
            }
        }
    }

    void Application::RenderDesktopUi() {
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

        // 1. Fullscreen transparent dockspace.
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        // 2. Top Main Menu Bar.
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit", "Alt+F4")) {
                    m_window->RequestClose();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("2D Orbital Viewport", nullptr, &m_showViewport2DPanel);
                ImGui::MenuItem("Simulation & Invariants", nullptr, &m_showSimulationPanel);
                ImGui::MenuItem("Multiverse Hierarchy", nullptr, &m_showHierarchyPanel);
                ImGui::MenuItem("Hardware & Adaptation", nullptr, &m_showHardwarePanel);
                ImGui::EndMenu();
            }

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

        // 3. Hardware & Adaptation Telemetry Panel.
        if (m_showHardwarePanel) {
            ImGui::SetNextWindowSize(ImVec2(440.0f, 480.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Hardware & Adaptation", &m_showHardwarePanel)) {
                ImGui::Text("Engine Version : v%s", kEngineVersion);
                ImGui::Text("Execution Mode : %s", m_adaptationPlan.executionModeLabel.c_str());
                ImGui::Text("Hardware Tier  : %s", SystemProfiler::HardwareTierToString(m_adaptationPlan.tier));
                ImGui::Separator();

                ImGui::Text("Host OS        : %s", m_hardwareCaps.osName.c_str());
                ImGui::Text("CPU            : %s", m_hardwareCaps.cpuModelName.c_str());
                ImGui::Text(
                    "CPU Threads    : %u total (%u adapted workers)",
                    m_hardwareCaps.cpuLogicalCores,
                    m_adaptationPlan.recommendedHostWorkerThreads
                );

                const double totalRamGiB = static_cast<double>(m_hardwareCaps.totalSystemRAMBytes) / kBytesPerGiB;
                const double availRamGiB = static_cast<double>(m_hardwareCaps.availableSystemRAMBytes) / kBytesPerGiB;
                const double usedRamGiB = (totalRamGiB > availRamGiB) ? (totalRamGiB - availRamGiB) : 0.0;
                const float ramFrac = (totalRamGiB > 0.0) ? static_cast<float>(usedRamGiB / totalRamGiB) : 0.0f;
                ImGui::Text("Host RAM       : %.2f / %.2f GiB used", usedRamGiB, totalRamGiB);
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
                    m_renderDevice->IsMatchedToCudaGpu() ? "UUID Matched to CUDA" : "Fallback"
                );
                if (m_cudaContext != nullptr && m_cudaContext->IsInitialized()) {
                    ImGui::Text(
                        "CUDA Context   : GPU [%d] | %zu streams | Tracked VRAM: %s / %s",
                        m_cudaContext->GetDeviceIndex(),
                        m_cudaContext->GetStreamCount(),
                        compute::FormatBytes(m_cudaContext->GetTrackedAllocatedBytes()).c_str(),
                        compute::FormatBytes(m_cudaContext->GetVRAMBudgetBytes()).c_str()
                    );
                }

                for (const auto& gpu : m_hardwareCaps.gpus) {
                    const double totalVramGiB = static_cast<double>(gpu.totalVRAMBytes) / kBytesPerGiB;
                    const double freeVramGiB = static_cast<double>(gpu.freeVRAMBytes) / kBytesPerGiB;
                    const double usedVramGiB = (totalVramGiB > freeVramGiB) ? (totalVramGiB - freeVramGiB) : 0.0;
                    const float vramFrac = (totalVramGiB > 0.0) ? static_cast<float>(usedVramGiB / totalVramGiB) : 0.0f;

                    ImGui::Spacing();
                    if (ImGui::CollapsingHeader(
                            (std::string("GPU [") + std::to_string(gpu.deviceIndex) + "] " + gpu.name).c_str(),
                            ImGuiTreeNodeFlags_DefaultOpen
                        )) {
                        ImGui::Text(
                            "  %s (SM %d.%d) | %d SMs",
                            gpu.architectureName.c_str(),
                            gpu.computeCapabilityMajor,
                            gpu.computeCapabilityMinor,
                            gpu.multiProcessorCount
                        );
                        ImGui::Text("  VRAM: %.2f / %.2f GiB used", usedVramGiB, totalVramGiB);
                        ImGui::ProgressBar(vramFrac, ImVec2(-1.0f, 0.0f));
                        ImGui::Text("  UUID: %s", SystemProfiler::FormatUuid(gpu.uuid).c_str());
                    }
                }
            }
            ImGui::End();
        }

        // 4. Multiverse Hierarchy Inspector (Live `.umgebung` Universes + Nested Bodies).
        if (m_showHierarchyPanel) {
            ImGui::SetNextWindowSize(ImVec2(360.0f, 440.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Multiverse Hierarchy", &m_showHierarchyPanel)) {
                if (ImGui::TreeNodeEx("Multiverse (Source)", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (m_multiverse.IsEmpty()) {
                        ImGui::TextDisabled("No .umgebung scenes loaded.");
                    } else {
                        const auto& universes = m_multiverse.GetUniverses();
                        for (std::size_t idx = 0U; idx < universes.size(); ++idx) {
                            const auto& u = universes[idx];
                            const bool isActive = (idx == m_multiverse.GetActiveUniverseIndex());

                            ImGui::PushID(static_cast<int>(idx));
                            if (ImGui::RadioButton("##select", isActive)) {
                                (void)ActivateUniverseForSimulation(idx);
                            }
                            ImGui::SameLine();

                            const std::string nodeLabel =
                                u.name + " [" + scene::PhysicsProfileTypeToString(u.physicsProfile.type) + "]";
                            if (ImGui::TreeNodeEx(
                                    nodeLabel.c_str(),
                                    isActive ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None
                                )) {
                                ImGui::TextDisabled(
                                    "ID: %s | Integrator: %s | Units: %s",
                                    u.universeId.c_str(),
                                    scene::IntegratorTypeToString(u.physicsProfile.integrator),
                                    u.physicsProfile.unitSystem.c_str()
                                );

                                // Render live body positions for the active universe, or initial state otherwise.
                                const auto& displayBodies = isActive ? m_liveBodies : u.bodies;
                                RenderBodyHierarchyNode(displayBodies, "");
                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                        }
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::End();
        }

        // 5. Simulation Control & Conservation Invariants Panel.
        if (m_showSimulationPanel) {
            ImGui::SetNextWindowSize(ImVec2(450.0f, 410.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Simulation & Invariants", &m_showSimulationPanel)) {
                const scene::UniverseScene* active = m_multiverse.GetActiveUniverse();
                if (active == nullptr || m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
                    ImGui::TextDisabled("Load a .umgebung scene to run CUDA physics simulation.");
                } else {
                    ImGui::Text("Active Universe : %s", active->name.c_str());
                    ImGui::Text("Active Solver   : %s", m_activeSolver->GetSolverName());
                    ImGui::Text(
                        "Physics Profile : %s (%s)",
                        scene::PhysicsProfileTypeToString(active->physicsProfile.type),
                        scene::IntegratorTypeToString(m_activeSolver->GetIntegratorType())
                    );
                    ImGui::Separator();

                    if (ImGui::Button(m_simulationRunning ? "Pause" : "Play")) {
                        m_simulationRunning = !m_simulationRunning;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Single Step")) {
                        m_simulationRunning = false;
                        if (m_activeSolver->StepMultiple(1ULL, m_activeTimestep)) {
                            (void)m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics);
                            (void)m_activeSolver->SyncToHost(m_liveBodies);
                            RecordLiveTrails();
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Reset")) {
                        (void)ActivateUniverseForSimulation(m_multiverse.GetActiveUniverseIndex());
                    }

                    ImGui::SliderInt("Steps / Frame", &m_stepsPerFrame, 1, 100);
                    ImGui::Text("Timestep (dt)   : %.6g | Bodies: %zu", m_activeTimestep, m_activeSolver->GetBodyCount());
                    ImGui::Separator();

                    ImGui::Text("Step Count      : %llu", static_cast<unsigned long long>(m_latestDiagnostics.stepCount));
                    ImGui::Text("Simulation Time : %.6f", m_latestDiagnostics.simulationTime);
                    ImGui::Text("Kinetic (T)     : %.10f", m_latestDiagnostics.kineticEnergy);
                    ImGui::Text("Potential (U)   : %.10f", m_latestDiagnostics.potentialEnergy);
                    ImGui::Text("Total Energy (E): %.10f", m_latestDiagnostics.totalEnergy);
                    ImGui::Text("Rel Energy Drift: %.3e", m_latestDiagnostics.relativeEnergyDrift);
                    ImGui::Text("|Momentum P|    : %.3e", m_latestDiagnostics.totalLinearMomentum.Norm());
                    ImGui::Text("|Angular Mom L| : %.10f", m_latestDiagnostics.totalAngularMomentum.Norm());
                    ImGui::Text(
                        "Center of Mass  : (%.3e, %.3e, %.3e)",
                        m_latestDiagnostics.centerOfMass.x,
                        m_latestDiagnostics.centerOfMass.y,
                        m_latestDiagnostics.centerOfMass.z
                    );

                    if (m_latestDiagnostics.hasUnifiedTelemetry) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "Pillar 2 Unified Field Telemetry:");
                        ImGui::Text("  Model           : %s", m_latestDiagnostics.solverSummary.c_str());
                        ImGui::Text("  <P / P0> Vacuum : %.8f", m_latestDiagnostics.meanVacuumPressureRatio);
                        ImGui::Text("  <c' / c> Light  : %.8f", m_latestDiagnostics.meanLocalLightSpeedRatio);
                        ImGui::Text("  <dTau / dt>     : %.6f", m_latestDiagnostics.meanLocalTimeRate);
                        ImGui::Text("  <Proper Time>   : %.6f", m_latestDiagnostics.meanLocalProperTime);
                        ImGui::Text("  <Haramein phi>  : %.4e", m_latestDiagnostics.meanHolographicPhi);
                        ImGui::Text("  Phryll Flux     : %.4g", m_latestDiagnostics.totalPhryllFlux);
                    }
                }
            }
            ImGui::End();
        }

        // 6. Live 2D Orbital Viewport (XY Plane Projection with Orbital Trails).
        if (m_showViewport2DPanel) {
            ImGui::SetNextWindowSize(ImVec2(540.0f, 540.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("2D Orbital Viewport (XY Plane)", &m_showViewport2DPanel)) {
                // Camera controls: Fixed scale (default) prevents escaping bodies
                // from pinning to one pixel while shrinking the central galaxy.
                ImGui::Checkbox("Auto-Fit", &m_viewportAutoFit);
                ImGui::SameLine();
                ImGui::BeginDisabled(m_viewportAutoFit);
                ImGui::SetNextItemWidth(160.0f);
                ImGui::SliderFloat("View Radius", &m_viewportHalfExtent, 0.25f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();

                double activeHalfExtent = static_cast<double>(m_viewportHalfExtent);
                if (m_viewportAutoFit) {
                    activeHalfExtent = 1.35;
                    for (const auto& b : m_liveBodies) {
                        activeHalfExtent = std::max(
                            activeHalfExtent,
                            std::max(std::abs(b.position.x), std::abs(b.position.y)) * 1.25
                        );
                    }
                    m_viewportHalfExtent = static_cast<float>(activeHalfExtent);
                }

                ImGui::SameLine();
                ImGui::TextDisabled("| Scale: +/-%.2f units", activeHalfExtent);

                const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
                const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
                if (canvasSize.x > 32.0f && canvasSize.y > 32.0f) {
                    const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
                    ImDrawList* drawList = ImGui::GetWindowDrawList();

                    // Clip all canvas lines/markers strictly inside the viewport rectangle.
                    drawList->PushClipRect(canvasMin, canvasMax, true);

                    // Space background, border, coordinate axes, and concentric reference rings.
                    drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(10, 14, 24, 255));
                    drawList->AddRect(canvasMin, canvasMax, IM_COL32(50, 70, 105, 255));

                    const ImVec2 center(
                        canvasMin.x + (0.5f * canvasSize.x),
                        canvasMin.y + (0.5f * canvasSize.y)
                    );
                    const float minCanvasDim = std::min(canvasSize.x, canvasSize.y);
                    const float pixelsPerUnit = (0.45f * minCanvasDim) / static_cast<float>(activeHalfExtent);

                    drawList->AddLine(
                        ImVec2(canvasMin.x, center.y),
                        ImVec2(canvasMax.x, center.y),
                        IM_COL32(40, 55, 80, 160)
                    );
                    drawList->AddLine(
                        ImVec2(center.x, canvasMin.y),
                        ImVec2(center.x, canvasMax.y),
                        IM_COL32(40, 55, 80, 160)
                    );
                    // Subtle reference rings at 50% and 100% of `activeHalfExtent`.
                    drawList->AddCircle(center, 0.225f * minCanvasDim, IM_COL32(40, 58, 86, 110), 64, 1.0f);
                    drawList->AddCircle(center, 0.450f * minCanvasDim, IM_COL32(40, 58, 86, 110), 64, 1.0f);

                    static constexpr ImU32 kPalette[6] = {
                        IM_COL32(255, 210, 90, 255),  // Gold (Star / Body 1)
                        IM_COL32(95, 190, 255, 255),  // Cyan (Planet / Body 2)
                        IM_COL32(255, 115, 145, 255), // Rose (Body 3)
                        IM_COL32(130, 245, 165, 255), // Mint
                        IM_COL32(195, 150, 255, 255), // Lavender
                        IM_COL32(255, 165, 85, 255)   // Amber
                    };

                    // Draw orbital trails first.
                    for (std::size_t i = 0U; i < m_bodyTrails.size(); ++i) {
                        const auto& trail = m_bodyTrails[i];
                        if (trail.size() < 2U) {
                            continue;
                        }
                        const ImU32 baseColor = kPalette[i % 6U];
                        const ImU32 trailColor = (baseColor & 0x00FFFFFFU) | (135U << 24U);
                        for (std::size_t k = 1U; k < trail.size(); ++k) {
                            const ImVec2 p0(
                                center.x + (static_cast<float>(trail[k - 1U].x) * pixelsPerUnit),
                                center.y - (static_cast<float>(trail[k - 1U].y) * pixelsPerUnit)
                            );
                            const ImVec2 p1(
                                center.x + (static_cast<float>(trail[k].x) * pixelsPerUnit),
                                center.y - (static_cast<float>(trail[k].y) * pixelsPerUnit)
                            );
                            drawList->AddLine(p0, p1, trailColor, 1.5f);
                        }
                    }

                    // Draw body markers on top of trails.
                    for (std::size_t i = 0U; i < m_liveBodies.size(); ++i) {
                        const auto& b = m_liveBodies[i];
                        const ImVec2 screenPos(
                            center.x + (static_cast<float>(b.position.x) * pixelsPerUnit),
                            center.y - (static_cast<float>(b.position.y) * pixelsPerUnit)
                        );
                        const ImU32 color = kPalette[i % 6U];
                        const float markerRadius = (b.category == "Star") ? 7.0f : 5.0f;
                        drawList->AddCircleFilled(screenPos, markerRadius, color);
                        drawList->AddText(
                            ImVec2(screenPos.x + markerRadius + 4.0f, screenPos.y - 7.0f),
                            IM_COL32(225, 232, 245, 230),
                            b.name.c_str()
                        );
                    }

                    drawList->PopClipRect();
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
            spdlog::info("Entering interactive Vulkan/NVRHI/ImGui render & CUDA simulation loop...");

            while (!m_window->ShouldClose()) {
                if (m_window->IsMinimized()) {
                    m_window->WaitEvents();
                    continue;
                }

                m_window->PollEvents();

                // Step the active GPU physics solver when playing, syncing live telemetry for ImGui.
                if (m_simulationRunning && m_activeSolver != nullptr && m_activeSolver->IsInitialized()) {
                    const std::uint64_t subSteps = static_cast<std::uint64_t>(std::max(1, m_stepsPerFrame));
                    if (m_activeSolver->StepMultiple(subSteps, m_activeTimestep)) {
                        (void)m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics);
                        (void)m_activeSolver->SyncToHost(m_liveBodies);
                        RecordLiveTrails();
                    }
                }

                if (m_renderDevice->BeginFrame()) {
                    RenderDesktopUi();
                    m_renderDevice->EndFrameAndPresent();
                }
            }

            spdlog::info("Window close requested; exiting interactive desktop loop.");
            return 0;
        }
#endif

        return RunHeadlessSimulation();
    }

    void Application::Shutdown() noexcept {
        if (!m_initialized && m_cudaContext == nullptr && m_activeSolver == nullptr
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
            && m_renderDevice == nullptr && m_window == nullptr
#endif
        ) {
            return;
        }

        spdlog::info("Shutting down Umgebung Application...");

        // Strict dependency-safe teardown order:
        //   1. Physics solvers (release tracked DeviceBuffer<T> VRAM before CUDAContext closes)
        //   2. CUDAContext (synchronize & destroy CUDA streams, verify zero VRAM leaks)
        //   3. RenderDevice (destroy ImGui, swapchain, VkSurfaceKHR before GLFW window closes)
        //   4. Window (destroy native OS window & terminate GLFW)
        if (m_activeSolver != nullptr) {
            m_activeSolver->Shutdown();
            m_activeSolver.reset();
        }
        if (m_cudaContext != nullptr) {
            m_cudaContext->Shutdown();
            m_cudaContext.reset();
        }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        if (m_renderDevice != nullptr) {
            m_renderDevice->Shutdown();
            m_renderDevice.reset();
        }
        if (m_window != nullptr) {
            m_window->Shutdown();
            m_window.reset();
        }
#endif

        m_multiverse.Clear();
        m_liveBodies.clear();
        m_guiActive = false;
        m_initialized = false;
        spdlog::info("Umgebung Application shutdown complete.");
    }

} // namespace umgebung::core
