/**
 * @file Application.cpp
 * @brief Implementation of `umgebung::core::Application` orchestrating the
 *        three-scope scene workspace (`Scenario`, `BubbleUniverse`,
 *        `FlowerOfLifeMultiverse`), composable three-tier `ModelRegistry`,
 *        simultaneous GPU Model A + Model B overlaid execution, headless HPC
 *        batch reporting, and the Vulkan/NVRHI/Dear ImGui desktop workspace.
 */

#include "umgebung/core/Application.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/ModelRegistry.hpp"

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
                ("s,scene", "Path to a .scenario.umgebung, .umgebung, .multiverse.umgebung, or directory", cxxopts::value<std::string>())
                ("m,model", "Override Primary Model A (e.g., Mainstream::Newtonian, Mainstream::DarkMatterNFW, Thinker::Essonne, Thinker::Haramein, Thinker::Pais, Thinker::Danaan, Thinker::VogelWillis, Unified::Synthesis)", cxxopts::value<std::string>())
                ("compare-model", "Enable simultaneous GPU Model B overlay comparison against Primary Model A", cxxopts::value<std::string>())
                ("steps", "Override step count for headless batch simulation", cxxopts::value<std::uint64_t>())
                ("dt", "Override integration timestep dt (scene time units)", cxxopts::value<double>())
                ("o,output", "Write headless simulation & A/B divergence telemetry JSON to path", cxxopts::value<std::string>())
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
            if (result.count("model") > 0U) {
                const std::string modelText = result["model"].as<std::string>();
                scene::ModelId parsedModel{};
                if (!scene::ModelIdFromString(modelText, parsedModel)) {
                    std::cerr << "Unknown --model '" << modelText
                              << "'. Valid models: Mainstream::Newtonian, Mainstream::DarkMatterNFW, "
                                 "Thinker::Essonne, Thinker::Haramein, Thinker::Pais, "
                                 "Thinker::Danaan, Thinker::VogelWillis, Unified::Synthesis."
                              << std::endl;
                    return ApplicationInitResult::Failure;
                }
                m_options.overridePrimaryModel = parsedModel;
            }
            if (result.count("compare-model") > 0U) {
                const std::string cmpText = result["compare-model"].as<std::string>();
                scene::ModelId parsedCmp{};
                if (!scene::ModelIdFromString(cmpText, parsedCmp)) {
                    std::cerr << "Unknown --compare-model '" << cmpText << "'." << std::endl;
                    return ApplicationInitResult::Failure;
                }
                m_options.overrideCompareModel = parsedCmp;
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

    bool Application::LoadInitialWorkspace(const char* argv0) {
        m_scenarios.Clear();
        m_universes.Clear();
        m_flowerMultiverse.Clear();

        // Case 1: Explicit `--scene <path>` supplied on CLI.
        if (m_options.scenePath.has_value()) {
            const std::filesystem::path& target = m_options.scenePath.value();
            std::error_code ec;

            if (std::filesystem::is_directory(target, ec)) {
                std::size_t totalLoaded = 0U;
                totalLoaded += m_scenarios.LoadDirectory(target / "scenarios");
                totalLoaded += m_universes.LoadDirectory(target / "universes");
                if (totalLoaded == 0U) {
                    totalLoaded += m_scenarios.LoadDirectory(target);
                }

                // Also discover any `.multiverse.umgebung` manifest inside `target/multiverses`.
                const std::filesystem::path mvDir = target / "multiverses";
                if (std::filesystem::is_directory(mvDir, ec)) {
                    for (const auto& entry : std::filesystem::directory_iterator(mvDir, ec)) {
                        const std::string fn = entry.path().filename().string();
                        if (entry.is_regular_file(ec) && fn.find(".multiverse.umgebung") != std::string::npos) {
                            std::string mvErr;
                            if (m_flowerMultiverse.LoadFromFile(entry.path(), mvErr)) {
                                ++totalLoaded;
                                break;
                            }
                        }
                    }
                }

                if (totalLoaded == 0U) {
                    spdlog::error("No valid .umgebung scenes found in directory '{}'.", target.string());
                    return false;
                }
                return true;
            }

            const std::string filename = target.filename().string();
            std::string errMsg;
            if (filename.find(".multiverse.umgebung") != std::string::npos) {
                if (!m_flowerMultiverse.LoadFromFile(target, errMsg)) {
                    spdlog::error("Failed loading Flower of Life Multiverse '{}': {}", target.string(), errMsg);
                    return false;
                }
                return true;
            }

            scene::UniverseScene loaded{};
            if (!scene::UniverseScene::LoadFromFile(target, loaded, errMsg)) {
                spdlog::error("Failed loading requested scene '{}': {}", target.string(), errMsg);
                return false;
            }

            if (loaded.scope == scene::SimulationScope::BubbleUniverse) {
                return m_universes.AddUniverse(std::move(loaded), errMsg);
            }
            return m_scenarios.AddUniverse(std::move(loaded), errMsg);
        }

        // Case 2: Auto-discover organized `scenes/{scenarios,universes,multiverses}` directories.
        std::vector<std::filesystem::path> searchRoots;
        if (argv0 != nullptr && argv0[0] != '\0') {
            std::error_code ec;
            const std::filesystem::path exePath = std::filesystem::weakly_canonical(argv0, ec);
            if (!ec && exePath.has_parent_path()) {
                searchRoots.push_back(exePath.parent_path() / "scenes");
            }
        }
        searchRoots.push_back(std::filesystem::path("scenes"));

        for (const auto& root : searchRoots) {
            std::error_code ec;
            if (!std::filesystem::is_directory(root, ec)) {
                continue;
            }

            const std::size_t scenCount = m_scenarios.LoadDirectory(root / "scenarios");
            const std::size_t univCount = m_universes.LoadDirectory(root / "universes");

            bool mvLoaded = false;
            const std::filesystem::path mvDir = root / "multiverses";
            if (std::filesystem::is_directory(mvDir, ec)) {
                std::vector<std::filesystem::path> manifests;
                for (const auto& entry : std::filesystem::directory_iterator(mvDir, ec)) {
                    const std::string fn = entry.path().filename().string();
                    if (entry.is_regular_file(ec) && fn.find(".multiverse.umgebung") != std::string::npos) {
                        manifests.push_back(entry.path());
                    }
                }
                std::sort(manifests.begin(), manifests.end());
                for (const auto& mf : manifests) {
                    std::string mvErr;
                    if (m_flowerMultiverse.LoadFromFile(mf, mvErr)) {
                        mvLoaded = true;
                        break;
                    }
                }
            }

            // Fallback: flat `.umgebung` files directly in `root`.
            if (scenCount == 0U && univCount == 0U && !mvLoaded) {
                const std::size_t flatCount = m_scenarios.LoadDirectory(root);
                if (flatCount > 0U) {
                    spdlog::info("Auto-discovered {} .umgebung scene(s) from '{}'.", flatCount, root.string());
                    return true;
                }
                continue;
            }

            spdlog::info(
                "Auto-discovered workspace from '{}': {} Scenarios, {} Bubble Universes, Flower of Life Multiverse={}.",
                root.string(),
                scenCount,
                univCount,
                mvLoaded ? "LOADED" : "none"
            );
            return true;
        }

        spdlog::info("No default scenes auto-discovered (pass --scene <path> to load one).");
        return true;
    }

    bool Application::ActivateSceneForSimulation(
        const scene::UniverseScene& scene,
        scene::SimulationScope activeScope,
        std::optional<scene::ModelId> explicitPrimaryModel,
        std::optional<scene::ModelId> explicitCompareModel
    ) {
        if (m_cudaContext == nullptr || !m_cudaContext->IsInitialized()) {
            return false;
        }

        if (m_compareSolver != nullptr) {
            m_compareSolver->Shutdown();
            m_compareSolver.reset();
        }
        if (m_activeSolver != nullptr) {
            m_activeSolver->Shutdown();
            m_activeSolver.reset();
        }

        m_activeScope = activeScope;
        m_activeScene = scene;
        m_hasActiveScene = true;
        m_activeTimestep = m_options.overrideTimestep.value_or(scene.physicsProfile.timestep);
        m_liveBodies = scene.bodies;
        m_compareBodies = scene.bodies;
        m_abMetrics = ABComparisonMetrics{};

        // Resolve Primary Model A: explicit override -> CLI `--model` -> scene `default_model`.
        m_primaryModelId = explicitPrimaryModel.value_or(
            m_options.overridePrimaryModel.value_or(scene.physicsProfile.defaultModel)
        );

        // Resolve Simultaneous Overlay Model B.
        if (explicitCompareModel.has_value()) {
            m_compareOverlayEnabled = true;
            m_compareModelId = explicitCompareModel.value();
        } else if (m_options.overrideCompareModel.has_value()) {
            m_compareOverlayEnabled = true;
            m_compareModelId = m_options.overrideCompareModel.value();
        } else if (!scene.physicsProfile.recommendedCompareModel.empty()) {
            scene::ModelId recModel{};
            if (scene::ModelIdFromString(scene.physicsProfile.recommendedCompareModel, recModel)) {
                m_compareModelId = recModel;
            }
        }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        m_bodyTrails.assign(m_liveBodies.size(), {});
        m_compareTrails.assign(m_liveBodies.size(), {});

        double initialMaxCoord = 1.0;
        for (const auto& b : m_liveBodies) {
            initialMaxCoord = std::max(
                initialMaxCoord,
                std::max(std::abs(b.position.x), std::abs(b.position.y))
            );
        }
        m_viewportHalfExtent = static_cast<float>(initialMaxCoord * 1.4);
#endif

        if (scene.bodies.empty()) {
            return true;
        }

        // Initialize Primary Solver (Model A on CUDA stream 0).
        m_activeSolver = models::ModelRegistry::CreateSolver(m_primaryModelId, 0U);
        if (m_activeSolver == nullptr || !m_activeSolver->Initialize(*m_cudaContext, m_activeScene)) {
            spdlog::error(
                "Failed initializing Primary Model {} for '{}'.",
                scene::ModelIdToString(m_primaryModelId),
                m_activeScene.name
            );
            m_activeSolver.reset();
            return false;
        }
        (void)m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics);
        (void)m_activeSolver->GetLocalProperTimes(m_primaryProperTimes);

        // Optionally initialize Simultaneous Comparison Overlay Solver (Model B on CUDA stream 1).
        if (m_compareOverlayEnabled) {
            m_compareSolver = models::ModelRegistry::CreateSolver(m_compareModelId, 1U);
            if (m_compareSolver == nullptr || !m_compareSolver->Initialize(*m_cudaContext, m_activeScene)) {
                spdlog::warn(
                    "Failed initializing Comparison Overlay Model {}; disabling overlay.",
                    scene::ModelIdToString(m_compareModelId)
                );
                m_compareSolver.reset();
                m_compareOverlayEnabled = false;
            } else {
                (void)m_compareSolver->ComputeSolverDiagnostics(m_compareDiagnostics);
                (void)m_compareSolver->GetLocalProperTimes(m_compareProperTimes);
                UpdateABComparisonMetrics();
            }
        }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        RecordLiveTrails();
#endif
        return true;
    }

    bool Application::StepActiveSolvers(std::uint64_t stepCount, double dt) {
        if (m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
            return false;
        }

        // Enqueue both Model A (stream 0) and Model B (stream 1) concurrently on the GPU.
        const bool hasCompare = (m_compareOverlayEnabled && m_compareSolver != nullptr && m_compareSolver->IsInitialized());
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!m_activeSolver->Step(dt)) {
                return false;
            }
            if (hasCompare && !m_compareSolver->Step(dt)) {
                return false;
            }
        }

        if (!m_activeSolver->Synchronize()) {
            return false;
        }
        if (hasCompare && !m_compareSolver->Synchronize()) {
            return false;
        }

        if (!m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics) ||
            !m_activeSolver->SyncToHost(m_liveBodies)) {
            return false;
        }
        (void)m_activeSolver->GetLocalProperTimes(m_primaryProperTimes);

        if (hasCompare) {
            if (!m_compareSolver->ComputeSolverDiagnostics(m_compareDiagnostics) ||
                !m_compareSolver->SyncToHost(m_compareBodies)) {
                return false;
            }
            (void)m_compareSolver->GetLocalProperTimes(m_compareProperTimes);
            UpdateABComparisonMetrics();
        } else {
            m_abMetrics.active = false;
        }

        return true;
    }

    void Application::UpdateABComparisonMetrics() {
        m_abMetrics = ABComparisonMetrics{};
        if (!m_compareOverlayEnabled || m_compareSolver == nullptr ||
            m_liveBodies.size() != m_compareBodies.size()) {
            return;
        }

        m_abMetrics.active = true;
        const std::size_t n = m_liveBodies.size();
        m_abMetrics.perBody.resize(n);

        double sumDeltaRSq = 0.0;
        for (std::size_t i = 0U; i < n; ++i) {
            const double dr = (m_compareBodies[i].position - m_liveBodies[i].position).Norm();
            const double dv = (m_compareBodies[i].velocity - m_liveBodies[i].velocity).Norm();
            const double tauA = (i < m_primaryProperTimes.size()) ? m_primaryProperTimes[i] : m_latestDiagnostics.simulationTime;
            const double tauB = (i < m_compareProperTimes.size()) ? m_compareProperTimes[i] : m_compareDiagnostics.simulationTime;
            const double dTau = std::abs(tauB - tauA);

            m_abMetrics.perBody[i] = {m_liveBodies[i].name, dr, dv, dTau};
            m_abMetrics.maxDeltaR = std::max(m_abMetrics.maxDeltaR, dr);
            m_abMetrics.maxDeltaV = std::max(m_abMetrics.maxDeltaV, dv);
            m_abMetrics.maxDeltaProperTime = std::max(m_abMetrics.maxDeltaProperTime, dTau);
            sumDeltaRSq += (dr * dr);
        }

        m_abMetrics.rmsDeltaR = (n > 0U) ? std::sqrt(sumDeltaRSq / static_cast<double>(n)) : 0.0;
        m_abMetrics.deltaTotalEnergy = m_compareDiagnostics.totalEnergy - m_latestDiagnostics.totalEnergy;
    }

    ApplicationInitResult Application::Initialize(int argc, char** argv) {
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

        // Stage 2: Hardware Discovery & Runtime Adaptation
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

        // Stage 3: CUDA Context Initialization
        m_cudaContext = std::make_unique<compute::CUDAContext>();
        if (!m_cudaContext->Initialize(m_hardwareCaps, m_adaptationPlan)) {
            spdlog::critical("Failed to initialize CUDAContext on adapted primary GPU.");
            m_cudaContext.reset();
            return ApplicationInitResult::Failure;
        }

        if (m_options.profileOnly) {
            m_initialized = true;
            return ApplicationInitResult::Ready;
        }

        // Stage 4: Load Three-Scope Workspace & Activate Initial Scene
        const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
        if (!LoadInitialWorkspace(argv0)) {
            Shutdown();
            return ApplicationInitResult::Failure;
        }

        if (!m_scenarios.IsEmpty()) {
            const scene::UniverseScene* initial = m_scenarios.GetActiveUniverse();
            if (initial != nullptr &&
                !ActivateSceneForSimulation(*initial, scene::SimulationScope::Scenario)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }
        } else if (!m_universes.IsEmpty()) {
            const scene::UniverseScene* initial = m_universes.GetActiveUniverse();
            if (initial != nullptr &&
                !ActivateSceneForSimulation(*initial, scene::SimulationScope::BubbleUniverse)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }
        } else if (!m_flowerMultiverse.IsEmpty()) {
            const auto* activeNode = m_flowerMultiverse.GetActiveNode();
            if (activeNode != nullptr && activeNode->loadedUniverse.has_value()) {
                if (!ActivateSceneForSimulation(
                        activeNode->loadedUniverse.value(),
                        scene::SimulationScope::FlowerOfLifeMultiverse
                    )) {
                    Shutdown();
                    return ApplicationInitResult::Failure;
                }
            }
        }

        // Stage 5: Desktop Window & Vulkan/NVRHI RenderDevice (GUI Builds Only)
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
        if (!m_hasActiveScene || m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
            spdlog::info("Running in Headless Compute mode with no active scene loaded.");
            return 0;
        }

        const std::uint64_t stepsToRun =
            m_options.overrideSteps.value_or(m_activeScene.physicsProfile.defaultSteps);
        const double dt = m_activeTimestep;
        const models::SolverDiagnostics initialDiag = m_latestDiagnostics;

        spdlog::info(
            "Headless Batch Simulation: '{}' [scope={}, {} bodies, Model A={}, Model B={}, steps={}, dt={:.6g}]",
            m_activeScene.name,
            scene::SimulationScopeToString(m_activeScope),
            m_activeSolver->GetBodyCount(),
            scene::ModelIdToString(m_primaryModelId),
            m_compareOverlayEnabled ? scene::ModelIdToString(m_compareModelId) : "OFF",
            stepsToRun,
            dt
        );
        spdlog::info(
            "  Initial Model A    -> E0={:.12g} (T={:.12g}, U={:.12g}) | |P0|={:.3e} | |L0|={:.12g}",
            initialDiag.totalEnergy,
            initialDiag.kineticEnergy,
            initialDiag.potentialEnergy,
            initialDiag.totalLinearMomentum.Norm(),
            initialDiag.totalAngularMomentum.Norm()
        );

        const auto tStart = std::chrono::steady_clock::now();
        if (!StepActiveSolvers(stepsToRun, dt)) {
            spdlog::error("Headless GPU simulation stepping failed.");
            return 1;
        }
        const auto tEnd = std::chrono::steady_clock::now();
        const double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        const double usPerStep = (stepsToRun > 0ULL) ? ((elapsedMs * 1000.0) / static_cast<double>(stepsToRun)) : 0.0;
        spdlog::info(
            "  Completed {} steps in {:.3f} ms ({:.2f} us/step) | t={:.6f}",
            m_latestDiagnostics.stepCount,
            elapsedMs,
            usPerStep,
            m_latestDiagnostics.simulationTime
        );
        spdlog::info(
            "  Final Model A      -> E ={:.12g} (rel drift={:.3e}) | |P|={:.3e} | |L|={:.12g}",
            m_latestDiagnostics.totalEnergy,
            m_latestDiagnostics.relativeEnergyDrift,
            m_latestDiagnostics.totalLinearMomentum.Norm(),
            m_latestDiagnostics.totalAngularMomentum.Norm()
        );
        if (m_latestDiagnostics.hasUnifiedTelemetry) {
            spdlog::info(
                "  Thinker Telemetry  -> <P/P0>={:.6f} | <c'/c>={:.6f} | <tau>={:.6f} | Pais m_eff/m0={:.6f} | VogelSat={:.4f}{}",
                m_latestDiagnostics.meanVacuumPressureRatio,
                m_latestDiagnostics.meanLocalLightSpeedRatio,
                m_latestDiagnostics.meanLocalProperTime,
                m_latestDiagnostics.paisInertialMassFactor,
                m_latestDiagnostics.vogelCrystalSaturationRatio,
                m_latestDiagnostics.vogelCrystalSinging ? " [SINGING!]" : ""
            );
        }
        if (m_abMetrics.active) {
            spdlog::info(
                "  Simultaneous A/B   -> Model A ({}) vs. Model B ({}): maxDeltaR={:.6g}, rmsDeltaR={:.6g}, maxDeltaV={:.6g}, deltaE={:.6g}, maxDeltaTau={:.6g}",
                scene::ModelIdToString(m_primaryModelId),
                scene::ModelIdToString(m_compareModelId),
                m_abMetrics.maxDeltaR,
                m_abMetrics.rmsDeltaR,
                m_abMetrics.maxDeltaV,
                m_abMetrics.deltaTotalEnergy,
                m_abMetrics.maxDeltaProperTime
            );
        }

        if (m_options.outputPath.has_value()) {
            nlohmann::json bodiesJson = nlohmann::json::array();
            for (const auto& b : m_liveBodies) {
                bodiesJson.push_back({
                    {"id", b.id},
                    {"name", b.name},
                    {"scale_tier", scene::ScaleTierToString(b.scaleTier)},
                    {"mass", b.mass},
                    {"position", {b.position.x, b.position.y, b.position.z}},
                    {"velocity", {b.velocity.x, b.velocity.y, b.velocity.z}}
                });
            }

            nlohmann::json abJson{
                {"active", m_abMetrics.active},
                {"compare_model", m_compareOverlayEnabled ? scene::ModelIdToString(m_compareModelId) : ""},
                {"max_delta_r", m_abMetrics.maxDeltaR},
                {"rms_delta_r", m_abMetrics.rmsDeltaR},
                {"max_delta_v", m_abMetrics.maxDeltaV},
                {"delta_total_energy", m_abMetrics.deltaTotalEnergy},
                {"max_delta_proper_time", m_abMetrics.maxDeltaProperTime}
            };

            nlohmann::json report{
                {"scope", scene::SimulationScopeToString(m_activeScope)},
                {"scene_id", m_activeScene.universeId},
                {"scene_name", m_activeScene.name},
                {"primary_model", scene::ModelIdToString(m_primaryModelId)},
                {"solver_name", m_activeSolver->GetSolverName()},
                {"integrator", scene::IntegratorTypeToString(m_activeSolver->GetIntegratorType())},
                {"steps", m_latestDiagnostics.stepCount},
                {"dt", dt},
                {"simulation_time", m_latestDiagnostics.simulationTime},
                {"wall_time_ms", elapsedMs},
                {"initial_energy", initialDiag.totalEnergy},
                {"final_energy", m_latestDiagnostics.totalEnergy},
                {"nfw_halo_potential_energy", m_latestDiagnostics.nfwHaloPotentialEnergy},
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
                    {"haramein_omega_z", m_latestDiagnostics.harameinSpacetimeTorqueOmegaZ},
                    {"pais_max_poynting_flux", m_latestDiagnostics.paisMaxPoyntingFlux},
                    {"pais_inertial_mass_factor", m_latestDiagnostics.paisInertialMassFactor},
                    {"total_phryll_flux", m_latestDiagnostics.totalPhryllFlux},
                    {"vogel_crystal_saturation_ratio", m_latestDiagnostics.vogelCrystalSaturationRatio},
                    {"vogel_crystal_singing", m_latestDiagnostics.vogelCrystalSinging}
                }},
                {"ab_comparison", std::move(abJson)},
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

        if (m_compareOverlayEnabled && m_compareBodies.size() == m_liveBodies.size()) {
            if (m_compareTrails.size() != m_compareBodies.size()) {
                m_compareTrails.assign(m_compareBodies.size(), {});
            }
            for (std::size_t i = 0U; i < m_compareBodies.size(); ++i) {
                auto& trail = m_compareTrails[i];
                trail.push_back(m_compareBodies[i].position);
                if (trail.size() > kMaxTrailPoints) {
                    trail.pop_front();
                }
            }
        }
    }

    void Application::RenderBodyHierarchyNode(
        const std::vector<scene::BodyEntity>& bodies,
        const std::string& parentId,
        bool showScaleTierTags
    ) {
        for (std::size_t i = 0U; i < bodies.size(); ++i) {
            const auto& body = bodies[i];
            if (body.parentId != parentId) {
                continue;
            }

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

            std::string label = body.name;
            if (showScaleTierTags) {
                label += std::string(" [") + scene::ScaleTierToString(body.scaleTier) + "]";
            } else {
                label += " [" + body.category + "]";
            }
            label += "##" + body.id;

            if (ImGui::TreeNodeEx(label.c_str(), flags)) {
                ImGui::TextDisabled(
                    "m=%.6g | r=(%.3f, %.3f, %.3f)",
                    body.mass,
                    body.position.x,
                    body.position.y,
                    body.position.z
                );
                if (hasChildren) {
                    RenderBodyHierarchyNode(bodies, body.id, showScaleTierTags);
                }
                ImGui::TreePop();
            }
        }
    }

    void Application::RenderFlowerOfLifeViewportCanvas() {
        const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
        const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        if (canvasSize.x <= 32.0f || canvasSize.y <= 32.0f) {
            return;
        }

        const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(canvasMin, canvasMax, true);

        // Deep cosmic Void backdrop.
        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(8, 10, 22, 255));
        drawList->AddRect(canvasMin, canvasMax, IM_COL32(75, 65, 130, 255));

        const ImVec2 center(canvasMin.x + 0.5f * canvasSize.x, canvasMin.y + 0.5f * canvasSize.y);
        const float minDim = std::min(canvasSize.x, canvasSize.y);
        const double bubbleR = std::max(0.1, m_flowerMultiverse.GetBubbleRadius());
        const float pixelsPerUnit = (0.145f * minDim) / static_cast<float>(bubbleR);

        // Outer enclosing boundary ring ("Body of Source" containing the 19-sphere Flower of Life).
        drawList->AddCircle(center, 3.05f * static_cast<float>(bubbleR) * pixelsPerUnit, IM_COL32(120, 95, 210, 120), 96, 2.0f);

        const auto& nodes = m_flowerMultiverse.GetNodes();
        for (std::size_t idx = 0U; idx < nodes.size(); ++idx) {
            const auto& node = nodes[idx];
            const ImVec2 pos(
                center.x + static_cast<float>(node.latticePosition.x) * pixelsPerUnit,
                center.y - static_cast<float>(node.latticePosition.y) * pixelsPerUnit
            );
            const float radiusPx = static_cast<float>(node.bubbleRadius) * pixelsPerUnit;
            const bool populated = node.loadedUniverse.has_value();
            const bool isSelected = (idx == m_flowerMultiverse.GetActiveNodeIndex());

            if (populated) {
                drawList->AddCircleFilled(
                    pos, radiusPx,
                    isSelected ? IM_COL32(75, 185, 255, 48) : IM_COL32(150, 115, 255, 36),
                    64
                );
            }
            const ImU32 ringCol = isSelected
                ? IM_COL32(110, 235, 255, 240)
                : (populated ? IM_COL32(255, 210, 115, 210) : IM_COL32(125, 115, 175, 110));
            drawList->AddCircle(pos, radiusPx, ringCol, 64, populated ? 2.2f : 1.1f);

            if (populated) {
                drawList->AddCircleFilled(pos, 5.0f, ringCol);
                drawList->AddText(
                    ImVec2(pos.x + 8.0f, pos.y - 8.0f),
                    IM_COL32(240, 245, 255, 245),
                    node.label.c_str()
                );
            }
        }

        drawList->AddText(
            ImVec2(canvasMin.x + 12.0f, canvasMin.y + 10.0f),
            IM_COL32(195, 180, 255, 230),
            "Elena Danaan Flower of Life Multiverse (Intersecting Bubble Universes & The Void)"
        );
        drawList->PopClipRect();
    }

    void Application::RenderOrbitalViewportCanvas() {
        ImGui::Checkbox("Auto-Fit", &m_viewportAutoFit);
        ImGui::SameLine();
        ImGui::BeginDisabled(m_viewportAutoFit);
        ImGui::SetNextItemWidth(150.0f);
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
            if (m_compareOverlayEnabled) {
                for (const auto& b : m_compareBodies) {
                    activeHalfExtent = std::max(
                        activeHalfExtent,
                        std::max(std::abs(b.position.x), std::abs(b.position.y)) * 1.25
                    );
                }
            }
            m_viewportHalfExtent = static_cast<float>(activeHalfExtent);
        }

        ImGui::SameLine();
        ImGui::TextDisabled("| Scale: +/-%.2f units", activeHalfExtent);

        const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
        const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        if (canvasSize.x <= 32.0f || canvasSize.y <= 32.0f) {
            return;
        }

        const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(canvasMin, canvasMax, true);

        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(10, 14, 24, 255));
        drawList->AddRect(canvasMin, canvasMax, IM_COL32(50, 70, 105, 255));

        const ImVec2 center(canvasMin.x + 0.5f * canvasSize.x, canvasMin.y + 0.5f * canvasSize.y);
        const float minCanvasDim = std::min(canvasSize.x, canvasSize.y);
        const float pixelsPerUnit = (0.45f * minCanvasDim) / static_cast<float>(activeHalfExtent);

        drawList->AddLine(ImVec2(canvasMin.x, center.y), ImVec2(canvasMax.x, center.y), IM_COL32(40, 55, 80, 160));
        drawList->AddLine(ImVec2(center.x, canvasMin.y), ImVec2(center.x, canvasMax.y), IM_COL32(40, 55, 80, 160));
        drawList->AddCircle(center, 0.225f * minCanvasDim, IM_COL32(40, 58, 86, 110), 64, 1.0f);
        drawList->AddCircle(center, 0.450f * minCanvasDim, IM_COL32(40, 58, 86, 110), 64, 1.0f);

        static constexpr ImU32 kPalette[6] = {
            IM_COL32(255, 210, 90, 255),
            IM_COL32(95, 190, 255, 255),
            IM_COL32(255, 115, 145, 255),
            IM_COL32(130, 245, 165, 255),
            IM_COL32(195, 150, 255, 255),
            IM_COL32(255, 165, 85, 255)
        };

        // 1. Model B Overlay Ghost Dashed Trails (drawn underneath Model A).
        if (m_compareOverlayEnabled) {
            for (std::size_t i = 0U; i < m_compareTrails.size(); ++i) {
                const auto& trail = m_compareTrails[i];
                if (trail.size() < 2U) {
                    continue;
                }
                const ImU32 ghostColor = (kPalette[i % 6U] & 0x00FFFFFFU) | (150U << 24U);
                for (std::size_t k = 1U; k < trail.size(); k += 2U) {
                    const ImVec2 p0(
                        center.x + static_cast<float>(trail[k - 1U].x) * pixelsPerUnit,
                        center.y - static_cast<float>(trail[k - 1U].y) * pixelsPerUnit
                    );
                    const ImVec2 p1(
                        center.x + static_cast<float>(trail[k].x) * pixelsPerUnit,
                        center.y - static_cast<float>(trail[k].y) * pixelsPerUnit
                    );
                    drawList->AddLine(p0, p1, ghostColor, 1.3f);
                }
            }
        }

        // 2. Primary Model A Solid Orbital Trails.
        for (std::size_t i = 0U; i < m_bodyTrails.size(); ++i) {
            const auto& trail = m_bodyTrails[i];
            if (trail.size() < 2U) {
                continue;
            }
            const ImU32 trailColor = (kPalette[i % 6U] & 0x00FFFFFFU) | (175U << 24U);
            for (std::size_t k = 1U; k < trail.size(); ++k) {
                const ImVec2 p0(
                    center.x + static_cast<float>(trail[k - 1U].x) * pixelsPerUnit,
                    center.y - static_cast<float>(trail[k - 1U].y) * pixelsPerUnit
                );
                const ImVec2 p1(
                    center.x + static_cast<float>(trail[k].x) * pixelsPerUnit,
                    center.y - static_cast<float>(trail[k].y) * pixelsPerUnit
                );
                drawList->AddLine(p0, p1, trailColor, 1.8f);
            }
        }

        // 3. Draw Primary Model A filled markers & Model B hollow ghost rings + divergence connector.
        for (std::size_t i = 0U; i < m_liveBodies.size(); ++i) {
            const auto& bA = m_liveBodies[i];
            const ImVec2 posA(
                center.x + static_cast<float>(bA.position.x) * pixelsPerUnit,
                center.y - static_cast<float>(bA.position.y) * pixelsPerUnit
            );
            const ImU32 color = kPalette[i % 6U];
            const float markerRadius = (bA.category == "Star" || bA.category == "GalacticCore") ? 7.0f : 5.0f;

            if (m_compareOverlayEnabled && i < m_compareBodies.size()) {
                const auto& bB = m_compareBodies[i];
                const ImVec2 posB(
                    center.x + static_cast<float>(bB.position.x) * pixelsPerUnit,
                    center.y - static_cast<float>(bB.position.y) * pixelsPerUnit
                );
                drawList->AddLine(posA, posB, IM_COL32(255, 255, 255, 85), 1.0f);
                drawList->AddCircle(posB, markerRadius + 2.5f, color, 24, 2.0f);
            }

            drawList->AddCircleFilled(posA, markerRadius, color);
            drawList->AddText(
                ImVec2(posA.x + markerRadius + 4.0f, posA.y - 7.0f),
                IM_COL32(225, 232, 245, 230),
                bA.name.c_str()
            );
        }

        // In-viewport Model A / Model B legend overlay.
        const std::string legendA = std::string("Model A (Solid) : ") + scene::ModelIdToString(m_primaryModelId);
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 8.0f), IM_COL32(160, 230, 255, 240), legendA.c_str());
        if (m_compareOverlayEnabled) {
            const std::string legendB = std::string("Model B (Dashed/Ring): ") + scene::ModelIdToString(m_compareModelId);
            drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 24.0f), IM_COL32(255, 210, 130, 240), legendB.c_str());
        }

        drawList->PopClipRect();
    }

    void Application::RenderDesktopUi() {
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit", "Alt+F4")) {
                    m_window->RequestClose();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("2D Viewport", nullptr, &m_showViewport2DPanel);
                ImGui::MenuItem("Simulation & A/B Comparison", nullptr, &m_showSimulationPanel);
                ImGui::MenuItem("Workspace & Hierarchy", nullptr, &m_showHierarchyPanel);
                ImGui::MenuItem("Hardware & Adaptation", nullptr, &m_showHardwarePanel);
                ImGui::EndMenu();
            }

            const ImGuiIO& io = ImGui::GetIO();
            const float frameMs = (io.Framerate > 0.0f) ? (1000.0f / io.Framerate) : 0.0f;
            ImGui::Separator();
            ImGui::Text(
                "  %.1f FPS (%.2f ms) | Swapchain %ux%u | Scope: %s",
                io.Framerate,
                frameMs,
                m_renderDevice->GetSwapchainWidth(),
                m_renderDevice->GetSwapchainHeight(),
                scene::SimulationScopeToString(m_activeScope)
            );
            ImGui::EndMainMenuBar();
        }

        // Hardware & Adaptation Panel
        if (m_showHardwarePanel) {
            ImGui::SetNextWindowSize(ImVec2(420.0f, 440.0f), ImGuiCond_FirstUseEver);
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
            }
            ImGui::End();
        }

        // Scope-Aware Workspace & Hierarchy Panel (Scenarios / Bubble Universes / Flower of Life Multiverse)
        if (m_showHierarchyPanel) {
            ImGui::SetNextWindowSize(ImVec2(390.0f, 480.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Workspace & Hierarchy", &m_showHierarchyPanel)) {
                if (ImGui::BeginTabBar("SimulationScopeTabs")) {
                    // Tab 1: Standalone Scenarios (NO Multiverse wrapper!)
                    if (ImGui::BeginTabItem("Scenarios")) {
                        if (m_scenarios.IsEmpty()) {
                            ImGui::TextDisabled("No standalone *.scenario.umgebung files loaded.");
                        } else {
                            const auto& scenarios = m_scenarios.GetUniverses();
                            for (std::size_t idx = 0U; idx < scenarios.size(); ++idx) {
                                const auto& sc = scenarios[idx];
                                const bool isSelected =
                                    (m_activeScope == scene::SimulationScope::Scenario) &&
                                    m_hasActiveScene && (m_activeScene.universeId == sc.universeId);

                                ImGui::PushID(static_cast<int>(idx));
                                if (ImGui::RadioButton("##selScen", isSelected)) {
                                    m_showFlowerOfLifeLatticeView = false;
                                    (void)m_scenarios.SetActiveUniverseIndex(idx);
                                    (void)ActivateSceneForSimulation(sc, scene::SimulationScope::Scenario);
                                }
                                ImGui::SameLine();
                                const std::string nodeLabel =
                                    sc.name + " [" + scene::ModelIdToString(sc.physicsProfile.defaultModel) + "]";
                                if (ImGui::TreeNodeEx(
                                        nodeLabel.c_str(),
                                        isSelected ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None
                                    )) {
                                    ImGui::TextDisabled("Scenario ID: %s | Units: %s", sc.universeId.c_str(), sc.physicsProfile.unitSystem.c_str());
                                    RenderBodyHierarchyNode(isSelected ? m_liveBodies : sc.bodies, "", false);
                                    ImGui::TreePop();
                                }
                                ImGui::PopID();
                            }
                        }
                        ImGui::EndTabItem();
                    }

                    // Tab 2: Multi-Scale Bubble Universes (displays ScaleTier tags)
                    if (ImGui::BeginTabItem("Bubble Universes")) {
                        if (m_universes.IsEmpty()) {
                            ImGui::TextDisabled("No Bubble Universe *.umgebung files loaded.");
                        } else {
                            const auto& univs = m_universes.GetUniverses();
                            for (std::size_t idx = 0U; idx < univs.size(); ++idx) {
                                const auto& u = univs[idx];
                                const bool isSelected =
                                    (m_activeScope == scene::SimulationScope::BubbleUniverse) &&
                                    m_hasActiveScene && (m_activeScene.universeId == u.universeId);

                                ImGui::PushID(static_cast<int>(1000U + idx));
                                if (ImGui::RadioButton("##selUniv", isSelected)) {
                                    m_showFlowerOfLifeLatticeView = false;
                                    (void)m_universes.SetActiveUniverseIndex(idx);
                                    (void)ActivateSceneForSimulation(u, scene::SimulationScope::BubbleUniverse);
                                }
                                ImGui::SameLine();
                                if (ImGui::TreeNodeEx(
                                        u.name.c_str(),
                                        isSelected ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None
                                    )) {
                                    ImGui::TextDisabled("Bubble Universe | Default: %s", scene::ModelIdToString(u.physicsProfile.defaultModel));
                                    RenderBodyHierarchyNode(isSelected ? m_liveBodies : u.bodies, "", true);
                                    ImGui::TreePop();
                                }
                                ImGui::PopID();
                            }
                        }
                        ImGui::EndTabItem();
                    }

                    // Tab 3: Elena Danaan's Flower of Life Multiverse
                    if (ImGui::BeginTabItem("Flower of Life")) {
                        if (m_flowerMultiverse.IsEmpty()) {
                            ImGui::TextDisabled("No .multiverse.umgebung manifest loaded.");
                        } else {
                            ImGui::TextWrapped("%s", m_flowerMultiverse.GetName().c_str());
                            const auto& vs = m_flowerMultiverse.GetVoidState();
                            ImGui::TextDisabled(
                                "The Void: conductivity=%.2f | Love=%.0f Hz | Phryll=%.0f",
                                vs.bridgeConductivity, vs.loveBindingFrequency, vs.sourcePhryllPotential
                            );
                            ImGui::Checkbox("Show Flower of Life Lattice in Viewport", &m_showFlowerOfLifeLatticeView);
                            ImGui::Separator();

                            const auto& nodes = m_flowerMultiverse.GetNodes();
                            for (std::size_t idx = 0U; idx < nodes.size(); ++idx) {
                                const auto& node = nodes[idx];
                                if (!node.loadedUniverse.has_value()) {
                                    continue;
                                }
                                const bool isSelected = (idx == m_flowerMultiverse.GetActiveNodeIndex() &&
                                                         m_activeScope == scene::SimulationScope::FlowerOfLifeMultiverse);
                                ImGui::PushID(static_cast<int>(2000U + idx));
                                if (ImGui::Button(isSelected ? "Active" : "Enter")) {
                                    (void)m_flowerMultiverse.SetActiveNodeIndex(idx);
                                    m_showFlowerOfLifeLatticeView = false;
                                    (void)ActivateSceneForSimulation(
                                        node.loadedUniverse.value(),
                                        scene::SimulationScope::FlowerOfLifeMultiverse
                                    );
                                }
                                ImGui::SameLine();
                                ImGui::Text(
                                    "[Ring %u] %s (%zu entities)",
                                    node.ringIndex,
                                    node.label.c_str(),
                                    node.loadedUniverse->bodies.size()
                                );
                                ImGui::PopID();
                            }
                            ImGui::TextDisabled("Total Lattice Nodes: %zu (19-Sphere Flower of Life)", nodes.size());
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
            ImGui::End();
        }

        // Simulation Controls, Live Model Registry Dropdowns & Simultaneous A/B Divergence Panel
        if (m_showSimulationPanel) {
            ImGui::SetNextWindowSize(ImVec2(480.0f, 520.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Simulation & A/B Comparison", &m_showSimulationPanel)) {
                if (!m_hasActiveScene || m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
                    ImGui::TextDisabled("Select a Scenario or Bubble Universe to run CUDA physics simulation.");
                } else {
                    ImGui::Text("Active Scene  : %s", m_activeScene.name.c_str());

                    const auto& allModels = models::ModelRegistry::GetAllModels();

                    // Live Primary Model A dropdown
                    const auto* curDescA = models::ModelRegistry::FindById(m_primaryModelId);
                    if (ImGui::BeginCombo("Primary Model A", curDescA ? curDescA->displayName : "Unknown")) {
                        for (const auto& desc : allModels) {
                            const bool selected = (desc.id == m_primaryModelId);
                            const std::string itemLabel = std::string("[") + desc.tierLabel + "] " + desc.displayName;
                            if (ImGui::Selectable(itemLabel.c_str(), selected)) {
                                if (desc.id != m_primaryModelId) {
                                    (void)ActivateSceneForSimulation(
                                        m_activeScene, m_activeScope, desc.id,
                                        m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                                    );
                                }
                            }
                            if (selected) {
                                ImGui::SetItemDefaultFocus();
                            }
                        }
                        ImGui::EndCombo();
                    }

                    // Simultaneous GPU Overlay Model B toggle & dropdown
                    bool overlayToggled = m_compareOverlayEnabled;
                    if (ImGui::Checkbox("Simultaneous GPU Overlay (Model B)", &overlayToggled)) {
                        m_compareOverlayEnabled = overlayToggled;
                        (void)ActivateSceneForSimulation(
                            m_activeScene, m_activeScope, m_primaryModelId,
                            m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                        );
                    }

                    if (m_compareOverlayEnabled) {
                        const auto* curDescB = models::ModelRegistry::FindById(m_compareModelId);
                        if (ImGui::BeginCombo("Overlay Model B", curDescB ? curDescB->displayName : "Unknown")) {
                            for (const auto& desc : allModels) {
                                const bool selected = (desc.id == m_compareModelId);
                                const std::string itemLabel = std::string("[") + desc.tierLabel + "] " + desc.displayName;
                                if (ImGui::Selectable(itemLabel.c_str(), selected)) {
                                    if (desc.id != m_compareModelId) {
                                        (void)ActivateSceneForSimulation(
                                            m_activeScene, m_activeScope, m_primaryModelId, desc.id
                                        );
                                    }
                                }
                                if (selected) {
                                    ImGui::SetItemDefaultFocus();
                                }
                            }
                            ImGui::EndCombo();
                        }
                    }

                    ImGui::Separator();

                    if (ImGui::Button(m_simulationRunning ? "Pause" : "Play")) {
                        m_simulationRunning = !m_simulationRunning;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Single Step")) {
                        m_simulationRunning = false;
                        if (StepActiveSolvers(1ULL, m_activeTimestep)) {
                            RecordLiveTrails();
                        }
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Reset")) {
                        (void)ActivateSceneForSimulation(
                            m_activeScene, m_activeScope, m_primaryModelId,
                            m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                        );
                    }

                    ImGui::SliderInt("Steps / Frame", &m_stepsPerFrame, 1, 100);
                    ImGui::Text(
                        "Step: %llu | t=%.5f | dt=%.4g | Bodies: %zu",
                        static_cast<unsigned long long>(m_latestDiagnostics.stepCount),
                        m_latestDiagnostics.simulationTime,
                        m_activeTimestep,
                        m_activeSolver->GetBodyCount()
                    );
                    ImGui::Separator();

                    ImGui::Text("Total Energy (A): %.9f (drift=%.2e)", m_latestDiagnostics.totalEnergy, m_latestDiagnostics.relativeEnergyDrift);
                    ImGui::Text("|P| = %.2e  |  |L| = %.9f", m_latestDiagnostics.totalLinearMomentum.Norm(), m_latestDiagnostics.totalAngularMomentum.Norm());
                    if (m_primaryModelId == scene::ModelId::MainstreamDarkMatterNFW) {
                        ImGui::Text("NFW Halo U_NFW  : %.9f", m_latestDiagnostics.nfwHaloPotentialEnergy);
                    }

                    if (m_latestDiagnostics.hasUnifiedTelemetry) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "Thinker / Unified Field Telemetry (Model A):");
                        ImGui::Text("  Essonne <P/P0> : %.8f | <c'/c>: %.8f", m_latestDiagnostics.meanVacuumPressureRatio, m_latestDiagnostics.meanLocalLightSpeedRatio);
                        ImGui::Text("  Danaan <dTau>  : %.5f | <Tau>: %.5f", m_latestDiagnostics.meanLocalTimeRate, m_latestDiagnostics.meanLocalProperTime);
                        ImGui::Text("  Haramein <phi> : %.3e | OmegaZ: %.4g", m_latestDiagnostics.meanHolographicPhi, m_latestDiagnostics.harameinSpacetimeTorqueOmegaZ);
                        ImGui::Text("  Pais S_max     : %.3e | m_eff/m0: %.6f", m_latestDiagnostics.paisMaxPoyntingFlux, m_latestDiagnostics.paisInertialMassFactor);
                        ImGui::Text("  Vogel Phryll   : %.4g | Sat Ratio: %.4f", m_latestDiagnostics.totalPhryllFlux, m_latestDiagnostics.vogelCrystalSaturationRatio);
                        if (m_latestDiagnostics.vogelCrystalSinging) {
                            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "  WARNING: CRYSTAL SATURATION / SINGING THRESHOLD EXCEEDED!");
                        }
                    }

                    // Live Quantitative A/B Divergence Table
                    if (m_abMetrics.active) {
                        ImGui::Separator();
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.82f, 0.4f, 1.0f),
                            "Live A/B Divergence: %s vs %s",
                            scene::ModelIdToString(m_primaryModelId),
                            scene::ModelIdToString(m_compareModelId)
                        );
                        ImGui::Text(
                            "Max dR: %.4e | RMS dR: %.4e | Max dV: %.4e | dE: %.4e | Max dTau: %.4e",
                            m_abMetrics.maxDeltaR,
                            m_abMetrics.rmsDeltaR,
                            m_abMetrics.maxDeltaV,
                            m_abMetrics.deltaTotalEnergy,
                            m_abMetrics.maxDeltaProperTime
                        );
                        if (ImGui::BeginTable("ABDivergenceTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                            ImGui::TableSetupColumn("Entity");
                            ImGui::TableSetupColumn("dR");
                            ImGui::TableSetupColumn("dV");
                            ImGui::TableSetupColumn("dTau");
                            ImGui::TableHeadersRow();
                            for (const auto& row : m_abMetrics.perBody) {
                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(row.name.c_str());
                                ImGui::TableSetColumnIndex(1); ImGui::Text("%.4e", row.deltaR);
                                ImGui::TableSetColumnIndex(2); ImGui::Text("%.4e", row.deltaV);
                                ImGui::TableSetColumnIndex(3); ImGui::Text("%.4e", row.deltaProperTime);
                            }
                            ImGui::EndTable();
                        }
                    }
                }
            }
            ImGui::End();
        }

        // 2D Viewport (Orbital A/B Overlay OR Flower of Life Multiverse Lattice)
        if (m_showViewport2DPanel) {
            ImGui::SetNextWindowSize(ImVec2(560.0f, 560.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("2D Viewport (XY Plane / Flower of Life)", &m_showViewport2DPanel)) {
                if (m_showFlowerOfLifeLatticeView && !m_flowerMultiverse.IsEmpty()) {
                    RenderFlowerOfLifeViewportCanvas();
                } else {
                    RenderOrbitalViewportCanvas();
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

                if (m_simulationRunning && m_activeSolver != nullptr && m_activeSolver->IsInitialized()) {
                    const std::uint64_t subSteps = static_cast<std::uint64_t>(std::max(1, m_stepsPerFrame));
                    if (StepActiveSolvers(subSteps, m_activeTimestep)) {
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
        if (!m_initialized && m_cudaContext == nullptr && m_activeSolver == nullptr && m_compareSolver == nullptr
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
            && m_renderDevice == nullptr && m_window == nullptr
#endif
        ) {
            return;
        }

        spdlog::info("Shutting down Umgebung Application...");

        // Strict dependency-safe teardown order:
        //   1. Both GPU Physics Solvers (Model B overlay & Model A primary)
        //   2. CUDAContext (streams & VRAM verification)
        //   3. RenderDevice (ImGui, swapchain, VkSurfaceKHR)
        //   4. Window (GLFW window & terminate)
        if (m_compareSolver != nullptr) {
            m_compareSolver->Shutdown();
            m_compareSolver.reset();
        }
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

        m_scenarios.Clear();
        m_universes.Clear();
        m_flowerMultiverse.Clear();
        m_liveBodies.clear();
        m_compareBodies.clear();
        m_hasActiveScene = false;
        m_initialized = false;
        m_guiActive = false;
    }

} // namespace umgebung::core
