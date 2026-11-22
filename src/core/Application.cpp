/**
 * @file Application.cpp
 * @brief Implementation of `umgebung::core::Application` orchestrating the
 *        three-scope scene workspace, domain-filtered `ModelRegistry`, explicit
 *        `SimulationDomain` execution (`NBodyDynamics`, `FieldGrid2D`,
 *        `RadialAndScaleProfile`), simultaneous GPU Model A + Model B overlays,
 *        headless HPC batch reporting, and domain-adaptive Dear ImGui viewports.
 */

#include "umgebung/core/Application.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/ModelRegistry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
    #include "umgebung/graphics/RenderDevice.hpp"
    #include "umgebung/graphics/Window.hpp"
    #include "umgebung/ui/UIContext.hpp"
    #include "umgebung/ui/UIManager.hpp"
#endif

namespace umgebung::core {

    namespace {

        /** @brief Formats the list of valid model canonical names for a `SimulationDomain`. */
        [[nodiscard]] std::string FormatCompatibleModelsForDomain(scene::SimulationDomain domain) {
            const auto compatible = models::ModelRegistry::GetModelsForDomain(domain);
            std::ostringstream oss;
            for (std::size_t i = 0U; i < compatible.size(); ++i) {
                if (i > 0U) {
                    oss << ", ";
                }
                oss << compatible[i].canonicalName;
            }
            return oss.str();
        }

    } // anonymous namespace

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
                ("m,model", "Override Primary Model A (filtered strictly by active SimulationDomain)", cxxopts::value<std::string>())
                ("compare-model", "Enable simultaneous GPU Model B overlay comparison (filtered by active SimulationDomain)", cxxopts::value<std::string>())
                ("steps", "Override step count for headless batch simulation", cxxopts::value<std::uint64_t>())
                ("dt", "Override integration timestep dt (scene time units)", cxxopts::value<double>())
                ("o,output", "Write headless simulation & A/B divergence telemetry JSON to path", cxxopts::value<std::string>())
                ("width", "Initial desktop window width in pixels (GUI builds)", cxxopts::value<std::uint32_t>()->default_value("1280"))
                ("height", "Initial desktop window height in pixels (GUI builds)", cxxopts::value<std::uint32_t>()->default_value("720"))
                ("x11", "Force X11/XWayland GLFW backend on Linux (default; enables multi-viewport panel detachment)")
                ("wayland", "Force native Wayland GLFW backend on Linux (disables detached multi-viewports)")
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
            m_options.forceWayland = (result.count("wayland") > 0U);
            m_options.windowWidth = result["width"].as<std::uint32_t>();
            m_options.windowHeight = result["height"].as<std::uint32_t>();
            m_options.explicitWindowSize = (result.count("width") > 0U || result.count("height") > 0U);
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
                    std::cerr << "Unknown --model '" << modelText << "'." << std::endl;
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
                std::cerr << "Invalid window dimensions." << std::endl;
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
        if (m_options.logLevel == "trace")      { spdlog::set_level(spdlog::level::trace); }
        else if (m_options.logLevel == "debug") { spdlog::set_level(spdlog::level::debug); }
        else if (m_options.logLevel == "warn")  { spdlog::set_level(spdlog::level::warn); }
        else if (m_options.logLevel == "error") { spdlog::set_level(spdlog::level::err); }
        else if (m_options.logLevel == "off")   { spdlog::set_level(spdlog::level::off); }
        else                                    { spdlog::set_level(spdlog::level::info); }
    }

    bool Application::LoadInitialWorkspace(const char* argv0) {
        m_scenarios.Clear();
        m_universes.Clear();
        m_flowerMultiverse.Clear();

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

            if (scenCount == 0U && univCount == 0U && !mvLoaded) {
                const std::size_t flatCount = m_scenarios.LoadDirectory(root);
                if (flatCount > 0U) {
                    return true;
                }
                continue;
            }

            spdlog::info(
                "Auto-discovered workspace from '{}': {} Scenarios, {} Bubble Universes, Flower of Life Multiverse={}.",
                root.string(), scenCount, univCount, mvLoaded ? "LOADED" : "none"
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
        std::optional<scene::ModelId> explicitCompareModel,
        bool applyCliOverrides
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
        m_probeWaveformA.clear();
        m_probeWaveformB.clear();
        m_primaryFieldSnap = models::FieldGrid2DSnapshot{};
        m_compareFieldSnap = models::FieldGrid2DSnapshot{};
        m_primaryRadialSnap = models::RadialProfileSnapshot{};
        m_compareRadialSnap = models::RadialProfileSnapshot{};
        m_abMetrics = ABComparisonMetrics{};

        const scene::SimulationDomain domain = scene.domain;

        // 1. Resolve Primary Model A and enforce strict SimulationDomain compatibility!
        scene::ModelId requestedPrimary = scene.physicsProfile.defaultModel;
        if (explicitPrimaryModel.has_value()) {
            requestedPrimary = explicitPrimaryModel.value();
        } else if (applyCliOverrides && m_options.overridePrimaryModel.has_value()) {
            requestedPrimary = m_options.overridePrimaryModel.value();
        }

        if (!models::ModelRegistry::IsModelCompatibleWithDomain(requestedPrimary, domain)) {
            if (applyCliOverrides && m_options.overridePrimaryModel.has_value()) {
                spdlog::error(
                    "Model '{}' is physically incompatible with scene '{}' (domain={}). Valid models for {}: {}",
                    scene::ModelIdToString(requestedPrimary),
                    scene.name,
                    scene::SimulationDomainToString(domain),
                    scene::SimulationDomainToString(domain),
                    FormatCompatibleModelsForDomain(domain)
                );
                return false;
            }
            requestedPrimary = models::ModelRegistry::GetDefaultModelForDomain(domain);
        }
        m_primaryModelId = requestedPrimary;

        // 2. Resolve Simultaneous Comparison Overlay Model B and enforce domain compatibility!
        if (explicitCompareModel.has_value()) {
            m_compareOverlayEnabled = true;
            m_compareModelId = explicitCompareModel.value();
        } else if (applyCliOverrides && m_options.overrideCompareModel.has_value()) {
            m_compareOverlayEnabled = true;
            m_compareModelId = m_options.overrideCompareModel.value();
        } else {
            // Pick a valid domain-compatible default for Model B (recommended_compare_model if valid).
            scene::ModelId candidateCompare = models::ModelRegistry::GetDefaultModelForDomain(domain);
            if (!scene.physicsProfile.recommendedCompareModel.empty()) {
                scene::ModelId rec{};
                if (scene::ModelIdFromString(scene.physicsProfile.recommendedCompareModel, rec) &&
                    models::ModelRegistry::IsModelCompatibleWithDomain(rec, domain)) {
                    candidateCompare = rec;
                }
            }
            if (candidateCompare == m_primaryModelId) {
                const auto compatibles = models::ModelRegistry::GetModelsForDomain(domain);
                for (const auto& d : compatibles) {
                    if (d.id != m_primaryModelId) {
                        candidateCompare = d.id;
                        break;
                    }
                }
            }
            m_compareModelId = candidateCompare;
        }

        if (m_compareOverlayEnabled && !models::ModelRegistry::IsModelCompatibleWithDomain(m_compareModelId, domain)) {
            if (applyCliOverrides && m_options.overrideCompareModel.has_value()) {
                spdlog::error(
                    "Compare model '{}' is physically incompatible with scene '{}' (domain={}). Valid models for {}: {}",
                    scene::ModelIdToString(m_compareModelId),
                    scene.name,
                    scene::SimulationDomainToString(domain),
                    scene::SimulationDomainToString(domain),
                    FormatCompatibleModelsForDomain(domain)
                );
                return false;
            }
            m_compareOverlayEnabled = false;
        }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        m_currentScenePath = scene.sourcePath;
        m_bodyTrails.assign(m_liveBodies.size(), {});
        m_compareTrails.assign(m_liveBodies.size(), {});
        m_fieldHeatmapMode = 0;
        m_simTimeAccumulator = 0.0;
        m_lastFrameExecutedSteps = 0;
        m_viewportPanX = 0.0f;
        m_viewportPanY = 0.0f;
        if (m_selectedBodyIndex.has_value() && m_selectedBodyIndex.value() >= m_liveBodies.size()) {
            m_selectedBodyIndex.reset();
        }

        double initialMaxCoord = 1.0;
        for (const auto& b : m_liveBodies) {
            initialMaxCoord = std::max(
                initialMaxCoord,
                std::max(std::abs(b.position.x), std::abs(b.position.y))
            );
        }
        m_viewportHalfExtent = static_cast<float>(initialMaxCoord * 1.4);
#endif

        if (domain == scene::SimulationDomain::NBodyDynamics && scene.bodies.empty()) {
            return true;
        }

        // 3. Initialize Primary Solver (Model A on CUDA stream 0).
        m_activeSolver = models::ModelRegistry::CreateSolver(m_primaryModelId, domain, 0U);
        if (m_activeSolver == nullptr || !m_activeSolver->Initialize(*m_cudaContext, m_activeScene)) {
            spdlog::error(
                "Failed initializing Primary Model {} for '{}' [domain={}].",
                scene::ModelIdToString(m_primaryModelId),
                m_activeScene.name,
                scene::SimulationDomainToString(domain)
            );
            m_activeSolver.reset();
            return false;
        }
        (void)m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics);

        // 4. Optionally initialize Simultaneous Overlay Solver (Model B on CUDA stream 1).
        if (m_compareOverlayEnabled) {
            m_compareSolver = models::ModelRegistry::CreateSolver(m_compareModelId, domain, 1U);
            if (m_compareSolver == nullptr || !m_compareSolver->Initialize(*m_cudaContext, m_activeScene)) {
                spdlog::warn(
                    "Failed initializing Comparison Overlay Model {}; disabling overlay.",
                    scene::ModelIdToString(m_compareModelId)
                );
                m_compareSolver.reset();
                m_compareOverlayEnabled = false;
            } else {
                (void)m_compareSolver->ComputeSolverDiagnostics(m_compareDiagnostics);
            }
        }

        RefreshDomainSnapshots();
        UpdateABComparisonMetrics();
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        RecordLiveTrails();
        UpdateWindowTitle();
#endif
        return true;
    }

    void Application::RefreshDomainSnapshots() {
        if (m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
            return;
        }
        const bool hasCompare = (m_compareOverlayEnabled && m_compareSolver != nullptr && m_compareSolver->IsInitialized());
        const scene::SimulationDomain domain = m_activeScene.domain;

        if (domain == scene::SimulationDomain::NBodyDynamics) {
            (void)m_activeSolver->SyncToHost(m_liveBodies);
            (void)m_activeSolver->GetLocalProperTimes(m_primaryProperTimes);
            if (hasCompare) {
                (void)m_compareSolver->SyncToHost(m_compareBodies);
                (void)m_compareSolver->GetLocalProperTimes(m_compareProperTimes);
            }
        } else if (domain == scene::SimulationDomain::FieldGrid2D) {
            (void)m_activeSolver->FetchFieldGrid2DSnapshot(m_primaryFieldSnap);
            m_probeWaveformA.push_back(static_cast<float>(m_primaryFieldSnap.probeSampleValue));
            if (m_probeWaveformA.size() > kMaxWaveformSamples) {
                m_probeWaveformA.pop_front();
            }
            if (hasCompare) {
                (void)m_compareSolver->FetchFieldGrid2DSnapshot(m_compareFieldSnap);
                m_probeWaveformB.push_back(static_cast<float>(m_compareFieldSnap.probeSampleValue));
                if (m_probeWaveformB.size() > kMaxWaveformSamples) {
                    m_probeWaveformB.pop_front();
                }
            }
        } else if (domain == scene::SimulationDomain::RadialAndScaleProfile) {
            (void)m_activeSolver->FetchRadialProfileSnapshot(m_primaryRadialSnap);
            if (hasCompare) {
                (void)m_compareSolver->FetchRadialProfileSnapshot(m_compareRadialSnap);
            }
        }
    }

    bool Application::StepActiveSolvers(std::uint64_t stepCount, double dt) {
        if (m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
            return false;
        }

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

        if (!m_activeSolver->ComputeSolverDiagnostics(m_latestDiagnostics)) {
            return false;
        }
        if (hasCompare && !m_compareSolver->ComputeSolverDiagnostics(m_compareDiagnostics)) {
            return false;
        }

        RefreshDomainSnapshots();
        UpdateABComparisonMetrics();
        return true;
    }

    void Application::UpdateABComparisonMetrics() {
        m_abMetrics = ABComparisonMetrics{};
        m_abMetrics.domain = m_activeScene.domain;
        if (!m_compareOverlayEnabled || m_compareSolver == nullptr) {
            return;
        }

        m_abMetrics.active = true;

        if (m_activeScene.domain == scene::SimulationDomain::NBodyDynamics) {
            if (m_liveBodies.size() != m_compareBodies.size()) {
                return;
            }
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
        } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D) {
            const auto& fA = m_primaryFieldSnap.scalarField;
            const auto& fB = m_compareFieldSnap.scalarField;
            m_abMetrics.fieldPeakA = m_primaryFieldSnap.peakAbsVal;
            m_abMetrics.fieldPeakB = m_compareFieldSnap.peakAbsVal;
            const std::size_t n = std::min(fA.size(), fB.size());
            double sumDiffSq = 0.0;
            double peakDiff = 0.0;
            for (std::size_t i = 0U; i < n; ++i) {
                const double d = std::abs(fB[i] - fA[i]);
                sumDiffSq += (d * d);
                peakDiff = std::max(peakDiff, d);
            }
            m_abMetrics.fieldL2Residual = (n > 0U) ? std::sqrt(sumDiffSq / static_cast<double>(n)) : 0.0;
            m_abMetrics.fieldPeakResidual = peakDiff;
            m_abMetrics.deltaTotalEnergy = m_compareDiagnostics.fieldEnergy - m_latestDiagnostics.fieldEnergy;
        } else if (m_activeScene.domain == scene::SimulationDomain::RadialAndScaleProfile) {
            const auto& yA = m_primaryRadialSnap.primaryYValues;
            const auto& yB = m_compareRadialSnap.primaryYValues;
            const std::size_t n = std::min(yA.size(), yB.size());
            double sumDiffSq = 0.0;
            double maxAbs = 0.0;
            double maxRel = 0.0;
            for (std::size_t i = 0U; i < n; ++i) {
                const double d = std::abs(yB[i] - yA[i]);
                const double denom = std::max({std::abs(yA[i]), std::abs(yB[i]), 1.0e-30});
                sumDiffSq += (d * d);
                maxAbs = std::max(maxAbs, d);
                maxRel = std::max(maxRel, d / denom);
            }
            m_abMetrics.curveMaxAbsDiff = maxAbs;
            m_abMetrics.curveRmsDiff = (n > 0U) ? std::sqrt(sumDiffSq / static_cast<double>(n)) : 0.0;
            m_abMetrics.curveMaxRelDiff = maxRel;
        }
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
            spdlog::critical("No usable NVIDIA CUDA GPU detected ({}).", m_hardwareCaps.cudaErrorMessage);
            return ApplicationInitResult::Failure;
        }

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

        const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
        if (!LoadInitialWorkspace(argv0)) {
            Shutdown();
            return ApplicationInitResult::Failure;
        }

        if (!m_scenarios.IsEmpty()) {
            const scene::UniverseScene* initial = m_scenarios.GetActiveUniverse();
            if (initial != nullptr &&
                !ActivateSceneForSimulation(*initial, scene::SimulationScope::Scenario, std::nullopt, std::nullopt, true)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }
        } else if (!m_universes.IsEmpty()) {
            const scene::UniverseScene* initial = m_universes.GetActiveUniverse();
            if (initial != nullptr &&
                !ActivateSceneForSimulation(*initial, scene::SimulationScope::BubbleUniverse, std::nullopt, std::nullopt, true)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }
        } else if (!m_flowerMultiverse.IsEmpty()) {
            const auto* activeNode = m_flowerMultiverse.GetActiveNode();
            if (activeNode != nullptr && activeNode->loadedUniverse.has_value()) {
                if (!ActivateSceneForSimulation(
                        activeNode->loadedUniverse.value(),
                        scene::SimulationScope::FlowerOfLifeMultiverse,
                        std::nullopt, std::nullopt, true
                    )) {
                    Shutdown();
                    return ApplicationInitResult::Failure;
                }
            }
        }

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
            // Maximize on startup by default (matching Umgebung-legacy's desktop
            // workstation experience) unless explicit --width/--height were requested.
            winConfig.maximized = !m_options.explicitWindowSize;
            winConfig.preferX11OnLinux = !m_options.forceWayland;
            winConfig.preferWaylandOnLinux = m_options.forceWayland && !m_options.forceX11;

            m_window = std::make_unique<graphics::Window>();
            if (!m_window->Initialize(winConfig)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }

            m_renderDevice = std::make_unique<graphics::RenderDevice>();
            if (!m_renderDevice->Initialize(*m_window, m_hardwareCaps, m_adaptationPlan)) {
                Shutdown();
                return ApplicationInitResult::Failure;
            }

            m_uiManager = std::make_unique<ui::UIManager>();
            m_uiManager->Initialize();
            UpdateWindowTitle();
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

        const auto* primaryDesc = models::ModelRegistry::FindById(m_primaryModelId);
        spdlog::info(
            "Headless Batch Simulation: '{}' [scope={}, domain={}, Model A={} ({}), Model B={}, steps={}, dt={:.6g}]",
            m_activeScene.name,
            scene::SimulationScopeToString(m_activeScope),
            scene::SimulationDomainToString(m_activeScene.domain),
            scene::ModelIdToString(m_primaryModelId),
            primaryDesc ? models::ModelProvenanceToString(primaryDesc->provenance) : "Unknown",
            m_compareOverlayEnabled ? scene::ModelIdToString(m_compareModelId) : "OFF",
            stepsToRun,
            dt
        );

        const auto tStart = std::chrono::steady_clock::now();
        if (!StepActiveSolvers(stepsToRun, dt)) {
            spdlog::error("Headless simulation stepping failed.");
            return 1;
        }
        const auto tEnd = std::chrono::steady_clock::now();
        const double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        spdlog::info(
            "  Completed {} steps in {:.3f} ms | t={:.6f}",
            m_latestDiagnostics.stepCount, elapsedMs, m_latestDiagnostics.simulationTime
        );

        if (m_activeScene.domain == scene::SimulationDomain::NBodyDynamics) {
            spdlog::info(
                "  Final N-Body (A)   -> E={:.12g} (rel drift={:.3e}) | |P|={:.3e} | |L|={:.12g}",
                m_latestDiagnostics.totalEnergy,
                m_latestDiagnostics.relativeEnergyDrift,
                m_latestDiagnostics.totalLinearMomentum.Norm(),
                m_latestDiagnostics.totalAngularMomentum.Norm()
            );
            if (m_abMetrics.active) {
                spdlog::info(
                    "  Simultaneous A/B   -> maxDeltaR={:.6g}, rmsDeltaR={:.6g}, maxDeltaV={:.6g}, deltaE={:.6g}, maxDeltaTau={:.6g}",
                    m_abMetrics.maxDeltaR, m_abMetrics.rmsDeltaR, m_abMetrics.maxDeltaV,
                    m_abMetrics.deltaTotalEnergy, m_abMetrics.maxDeltaProperTime
                );
            }
        } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D) {
            if (m_latestDiagnostics.hasC204Telemetry) {
                spdlog::info(
                    "  Final C204 Unit (A)-> State='{}' | Dir={} | Year={:.3f} (dT={:+.3f} yr) | Power={:.0f}% ({:.0f} C, {:.2f} yrs/hr, v_t={:.0f} s/s) | Cabin={:.2f} G | Temp={:.1f} F | Horizons(Neg/Null/Pos)={}/{}/{} | VGL={}",
                    m_latestDiagnostics.c204OperationalState,
                    m_latestDiagnostics.c204IsForwardTravel ? "FORWARD(-25%E)" : "BACKWARD(CTC)",
                    m_latestDiagnostics.c204CurrentWorldlineYear,
                    m_latestDiagnostics.c204ElapsedDisplacedYears,
                    m_latestDiagnostics.c204PowerLevelPercent,
                    m_latestDiagnostics.c204InjectedChargeCoulombs,
                    m_latestDiagnostics.c204PenetrationRateYrsPerHr,
                    m_latestDiagnostics.c204TimeVelocitySecPerSec,
                    m_latestDiagnostics.c204CabinGForce,
                    m_latestDiagnostics.c204ExteriorTempF,
                    m_latestDiagnostics.c204NegativeHorizonCells,
                    m_latestDiagnostics.c204NullHorizonCells,
                    m_latestDiagnostics.c204PositiveHorizonCells,
                    m_latestDiagnostics.c204VglCheckCount
                );
            } else {
                spdlog::info(
                    "  Final Field2D (A)  -> E_field={:.6g} | L2={:.6g} | Peak={:.6g} | CFL={:.4f} | Pais m_eff/m0={:.6f} | VogelSat={:.4f}{}",
                    m_latestDiagnostics.fieldEnergy,
                    m_latestDiagnostics.fieldL2Norm,
                    m_latestDiagnostics.fieldPeakValue,
                    m_latestDiagnostics.effectiveCflNumber,
                    m_latestDiagnostics.paisInertialMassFactor,
                    m_latestDiagnostics.vogelCrystalSaturationRatio,
                    m_latestDiagnostics.vogelCrystalSinging ? " [SINGING!]" : ""
                );
            }
            if (m_abMetrics.active) {
                spdlog::info(
                    "  Simultaneous A/B   -> fieldL2Residual={:.6g}, fieldPeakResidual={:.6g}, peakA={:.6g}, peakB={:.6g}",
                    m_abMetrics.fieldL2Residual, m_abMetrics.fieldPeakResidual,
                    m_abMetrics.fieldPeakA, m_abMetrics.fieldPeakB
                );
            }
        } else if (m_activeScene.domain == scene::SimulationDomain::RadialAndScaleProfile) {
            spdlog::info(
                "  Radial/Scale (A)   -> samples={} | <P/P0>={:.6f} | <c'/c>={:.6f} | <phi>={:.4e}",
                m_primaryRadialSnap.xValues.size(),
                m_latestDiagnostics.meanVacuumPressureRatio,
                m_latestDiagnostics.meanLocalLightSpeedRatio,
                m_latestDiagnostics.meanHolographicPhi
            );
            if (m_abMetrics.active) {
                spdlog::info(
                    "  Simultaneous A/B   -> curveMaxAbsDiff={:.6g}, curveRmsDiff={:.6g}, curveMaxRelDiff={:.6g}",
                    m_abMetrics.curveMaxAbsDiff, m_abMetrics.curveRmsDiff, m_abMetrics.curveMaxRelDiff
                );
            }
        }

        if (m_options.outputPath.has_value()) {
            nlohmann::json report{
                {"scope", scene::SimulationScopeToString(m_activeScope)},
                {"domain", scene::SimulationDomainToString(m_activeScene.domain)},
                {"scene_id", m_activeScene.universeId},
                {"scene_name", m_activeScene.name},
                {"primary_model", scene::ModelIdToString(m_primaryModelId)},
                {"primary_model_provenance", primaryDesc ? models::ModelProvenanceToString(primaryDesc->provenance) : "Unknown"},
                {"primary_model_provenance_details", primaryDesc ? primaryDesc->provenanceDetails : ""},
                {"compare_model", m_compareOverlayEnabled ? scene::ModelIdToString(m_compareModelId) : ""},
                {"steps", m_latestDiagnostics.stepCount},
                {"dt", dt},
                {"simulation_time", m_latestDiagnostics.simulationTime},
                {"wall_time_ms", elapsedMs},
                {"final_energy", m_latestDiagnostics.totalEnergy},
                {"relative_energy_drift", m_latestDiagnostics.relativeEnergyDrift},
                {"field_l2_norm", m_latestDiagnostics.fieldL2Norm},
                {"field_peak_value", m_latestDiagnostics.fieldPeakValue},
                {"ab_comparison", {
                    {"active", m_abMetrics.active},
                    {"max_delta_r", m_abMetrics.maxDeltaR},
                    {"rms_delta_r", m_abMetrics.rmsDeltaR},
                    {"field_l2_residual", m_abMetrics.fieldL2Residual},
                    {"field_peak_residual", m_abMetrics.fieldPeakResidual},
                    {"curve_max_abs_diff", m_abMetrics.curveMaxAbsDiff},
                    {"curve_rms_diff", m_abMetrics.curveRmsDiff}
                }},
                {"hardware", SystemProfiler::ToJson(m_hardwareCaps, m_adaptationPlan)}
            };

            if (m_latestDiagnostics.hasC204Telemetry) {
                report["titor_c204"] = {
                    {"operational_state", m_latestDiagnostics.c204OperationalState},
                    {"power_level_percent", m_latestDiagnostics.c204PowerLevelPercent},
                    {"injected_charge_coulombs", m_latestDiagnostics.c204InjectedChargeCoulombs},
                    {"expended_charge_coulombs", m_latestDiagnostics.c204ExpendedChargeCoulombs},
                    {"penetration_rate_yrs_per_hr", m_latestDiagnostics.c204PenetrationRateYrsPerHr},
                    {"time_velocity_sec_per_sec", m_latestDiagnostics.c204TimeVelocitySecPerSec},
                    {"current_worldline_year", m_latestDiagnostics.c204CurrentWorldlineYear},
                    {"elapsed_displaced_years", m_latestDiagnostics.c204ElapsedDisplacedYears},
                    {"worldline_divergence_pct", m_latestDiagnostics.c204WorldlineDivergencePct},
                    {"cabin_g_force", m_latestDiagnostics.c204CabinGForce},
                    {"exterior_temp_f", m_latestDiagnostics.c204ExteriorTempF},
                    {"metric_g00_at_cabin", m_latestDiagnostics.c204MetricG00AtCabin},
                    {"metric_grad_g00_at_cabin", m_latestDiagnostics.c204MetricGradG00AtCabin},
                    {"optical_phase", m_latestDiagnostics.c204OpticalPhase},
                    {"vgl_check_count", m_latestDiagnostics.c204VglCheckCount},
                    {"vgl_next_check_countdown_sec", m_latestDiagnostics.c204VglNextCheckCountdownSec},
                    {"borda_consensus_spread", m_latestDiagnostics.c204BordaConsensusSpread},
                    {"cabin_oxygen_pct", m_latestDiagnostics.c204CabinOxygenPct},
                    {"carbon_sensor_alert", m_latestDiagnostics.c204CarbonSensorAlert},
                    {"uv_burst_active", m_latestDiagnostics.c204UvBurstActive},
                    {"optical_blackout_active", m_latestDiagnostics.c204OpticalBlackoutActive},
                    {"origin_return_triggered", m_latestDiagnostics.c204OriginReturnTriggered},
                    {"abort_reason", m_latestDiagnostics.c204AbortReason},
                    {"is_forward_travel", m_latestDiagnostics.c204IsForwardTravel},
                    {"negative_horizon_cells", m_latestDiagnostics.c204NegativeHorizonCells},
                    {"null_horizon_cells", m_latestDiagnostics.c204NullHorizonCells},
                    {"positive_horizon_cells", m_latestDiagnostics.c204PositiveHorizonCells},
                    {"kerr_prograde_photon_r", m_latestDiagnostics.c204KerrProgradePhotonR},
                    {"kerr_retrograde_photon_r", m_latestDiagnostics.c204KerrRetrogradePhotonR}
                };
            }

            std::ofstream outFile(m_options.outputPath.value());
            if (!outFile.is_open()) {
                spdlog::error("Failed opening --output path '{}'.", m_options.outputPath->string());
                return 1;
            }
            outFile << report.dump(2) << '\n';
        }

        return 0;
    }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
    void Application::RecordLiveTrails() {
        if (m_activeScene.domain != scene::SimulationDomain::NBodyDynamics) {
            return;
        }
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

    void Application::UpdateWindowTitle() {
        if (m_window == nullptr || !m_window->IsInitialized()) {
            return;
        }
        const auto state = static_cast<ui::SimulationPlaybackState>(m_playbackStateRaw);
        std::string title = std::string("Umgebung v") + kEngineVersion +
                            " [" + ui::SimulationPlaybackStateToString(state) + "]";
        if (m_hasActiveScene && !m_activeScene.name.empty()) {
            title += " | " + m_activeScene.name;
        }
        if (m_cudaContext != nullptr && m_cudaContext->IsInitialized()) {
            const auto& primaryGpu = m_cudaContext->GetPrimaryGpuInfo();
            title += " | " + primaryGpu.name +
                     " (SM " + std::to_string(primaryGpu.computeCapabilityMajor) + "." +
                     std::to_string(primaryGpu.computeCapabilityMinor) + ")";
        }
        m_window->SetTitle(title);
    }

    void Application::CreateNewEmptyScene() {
        scene::UniverseScene blank{};
        blank.schemaVersion = scene::UniverseScene::kCurrentSchemaVersion;
        blank.scope = scene::SimulationScope::Scenario;
        blank.domain = scene::SimulationDomain::NBodyDynamics;
        blank.universeId = "custom_sandbox";
        blank.name = "Untitled Sandbox";
        blank.description = "Interactive N-body sandbox created from File -> New Scene.";
        blank.physicsProfile.type = scene::PhysicsProfileType::Mainstream;
        blank.physicsProfile.defaultModel = scene::ModelId::MainstreamNewtonian;
        blank.physicsProfile.mainstreamGravity.gravitationalConstant = 1.0;
        blank.physicsProfile.mainstreamGravity.softeningLength = 1.0e-3;
        blank.physicsProfile.timestep = 1.0e-3;
        blank.physicsProfile.defaultSteps = 1000ULL;

        scene::BodyEntity centralStar{};
        centralStar.id = "central_star";
        centralStar.name = "Central Star";
        centralStar.category = "Star";
        centralStar.scaleTier = scene::ScaleTier::StarSystem;
        centralStar.mass = 1.0;
        centralStar.radius = 0.25;
        centralStar.position = {0.0, 0.0, 0.0};
        centralStar.velocity = {0.0, 0.0, 0.0};
        centralStar.vibrationalDensity = 3.0;
        blank.bodies.push_back(std::move(centralStar));

        m_currentScenePath.clear();
        m_selectedBodyIndex.reset();
        m_showFlowerOfLifeLatticeView = false;
        m_playbackStateRaw = static_cast<std::uint8_t>(ui::SimulationPlaybackState::Stopped);
        (void)ActivateSceneForSimulation(blank, scene::SimulationScope::Scenario);
        spdlog::info("Created new interactive sandbox scene '{}'.", m_activeScene.name);
    }

    bool Application::OpenSceneFromPath(const std::filesystem::path& filePath) {
        const std::string filename = filePath.filename().string();
        std::string errMsg;

        if (filename.find(".multiverse.umgebung") != std::string::npos) {
            scene::FlowerOfLifeMultiverse loadedMv{};
            if (!loadedMv.LoadFromFile(filePath, errMsg)) {
                spdlog::error("Failed opening Flower of Life Multiverse '{}': {}", filePath.string(), errMsg);
                return false;
            }
            m_flowerMultiverse = std::move(loadedMv);
            m_showFlowerOfLifeLatticeView = true;
            if (const auto* activeNode = m_flowerMultiverse.GetActiveNode();
                activeNode != nullptr && activeNode->loadedUniverse.has_value()) {
                (void)ActivateSceneForSimulation(
                    activeNode->loadedUniverse.value(),
                    scene::SimulationScope::FlowerOfLifeMultiverse
                );
            }
            spdlog::info("Opened Flower of Life Multiverse '{}'.", filePath.string());
            return true;
        }

        scene::UniverseScene loaded{};
        if (!scene::UniverseScene::LoadFromFile(filePath, loaded, errMsg)) {
            spdlog::error("Failed opening scene '{}': {}", filePath.string(), errMsg);
            return false;
        }

        m_showFlowerOfLifeLatticeView = false;
        m_selectedBodyIndex.reset();
        const scene::SimulationScope scope = loaded.scope;
        if (!ActivateSceneForSimulation(loaded, scope)) {
            return false;
        }
        m_currentScenePath = filePath;
        spdlog::info("Opened scene '{}' from '{}'.", m_activeScene.name, filePath.string());
        return true;
    }

    bool Application::SaveActiveSceneToPath(const std::filesystem::path& filePath) {
        if (!m_hasActiveScene) {
            return false;
        }
        std::string errMsg;
        if (!m_activeScene.SaveToFile(filePath, errMsg)) {
            spdlog::error("Failed saving scene '{}' to '{}': {}", m_activeScene.name, filePath.string(), errMsg);
            return false;
        }
        m_currentScenePath = filePath;
        m_activeScene.sourcePath = filePath;
        spdlog::info("Saved scene '{}' to '{}'.", m_activeScene.name, filePath.string());
        return true;
    }

    void Application::AddBodyToActiveScene(const scene::BodyEntity& entity) {
        if (!m_hasActiveScene || m_activeScene.domain == scene::SimulationDomain::RadialAndScaleProfile) {
            return;
        }
        m_activeScene.bodies.push_back(entity);
        m_selectedBodyIndex = m_activeScene.bodies.size() - 1U;
        m_entitySelectionJustChanged = true;
        (void)ActivateSceneForSimulation(
            m_activeScene,
            m_activeScope,
            m_primaryModelId,
            m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
        );
        spdlog::info("Spawned entity '{}' [{}] in '{}'.", entity.name, entity.category, m_activeScene.name);
    }

    void Application::DeleteBodyFromActiveScene(std::size_t bodyIndex) {
        if (!m_hasActiveScene || bodyIndex >= m_activeScene.bodies.size()) {
            return;
        }
        const std::string removedName = m_activeScene.bodies[bodyIndex].name;
        m_activeScene.bodies.erase(m_activeScene.bodies.begin() + static_cast<std::ptrdiff_t>(bodyIndex));
        m_selectedBodyIndex.reset();
        (void)ActivateSceneForSimulation(
            m_activeScene,
            m_activeScope,
            m_primaryModelId,
            m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
        );
        spdlog::info("Deleted entity '{}' from '{}'.", removedName, m_activeScene.name);
    }

    void Application::FocusViewportOnBody(std::size_t bodyIndex) {
        if (!m_hasActiveScene || m_activeScene.domain != scene::SimulationDomain::NBodyDynamics) {
            return;
        }
        const auto& bodies = (bodyIndex < m_liveBodies.size()) ? m_liveBodies : m_activeScene.bodies;
        if (bodyIndex >= bodies.size()) {
            return;
        }
        m_viewportAutoFit = false;
        m_viewportPanX = static_cast<float>(bodies[bodyIndex].position.x);
        m_viewportPanY = static_cast<float>(bodies[bodyIndex].position.y);
        spdlog::debug("Focused viewport on '{}' at ({:.3f}, {:.3f}).", bodies[bodyIndex].name, m_viewportPanX, m_viewportPanY);
    }

    void Application::RenderDesktopUi() {
        if (m_uiManager == nullptr || !m_uiManager->IsInitialized()) {
            return;
        }

        ui::UIContext ctx{};
        ctx.engineVersion = kEngineVersion;
        ctx.hardwareCaps = &m_hardwareCaps;
        ctx.adaptationPlan = &m_adaptationPlan;
        ctx.cudaContext = m_cudaContext.get();
        if (m_renderDevice != nullptr) {
            ctx.selectedVulkanGpuName = m_renderDevice->GetSelectedVulkanGpuName();
            ctx.vulkanMatchedCudaUuid = m_renderDevice->IsMatchedToCudaGpu();
        }

        ctx.scenarios = &m_scenarios;
        ctx.universes = &m_universes;
        ctx.flowerMultiverse = &m_flowerMultiverse;

        ctx.activeScope = m_activeScope;
        ctx.hasActiveScene = m_hasActiveScene;
        ctx.activeScene = m_hasActiveScene ? &m_activeScene : nullptr;
        ctx.currentScenePath = m_currentScenePath;

        ctx.hasActiveSolver = (m_activeSolver != nullptr && m_activeSolver->IsInitialized());
        ctx.primaryModelId = m_primaryModelId;
        ctx.compareOverlayEnabled = m_compareOverlayEnabled;
        ctx.compareModelId = m_compareModelId;

        ctx.playbackState = static_cast<ui::SimulationPlaybackState>(m_playbackStateRaw);
        ctx.activeTimestep = &m_activeTimestep;
        ctx.wallClockPaced = &m_wallClockPaced;
        ctx.simTimeScale = &m_simTimeScale;
        ctx.simTimeAccumulator = &m_simTimeAccumulator;
        ctx.maxSubStepsPerFrame = m_maxSubStepsPerFrame;
        ctx.lastFrameExecutedSteps = m_lastFrameExecutedSteps;
        ctx.stepsPerFrame = &m_stepsPerFrame;

        ctx.liveBodies = &m_liveBodies;
        ctx.compareBodies = &m_compareBodies;
        ctx.primaryProperTimes = &m_primaryProperTimes;
        ctx.compareProperTimes = &m_compareProperTimes;
        ctx.bodyTrails = &m_bodyTrails;
        ctx.compareTrails = &m_compareTrails;

        ctx.primaryFieldSnap = &m_primaryFieldSnap;
        ctx.compareFieldSnap = &m_compareFieldSnap;
        ctx.probeWaveformA = &m_probeWaveformA;
        ctx.probeWaveformB = &m_probeWaveformB;

        ctx.primaryRadialSnap = &m_primaryRadialSnap;
        ctx.compareRadialSnap = &m_compareRadialSnap;

        ctx.latestDiagnostics = &m_latestDiagnostics;
        ctx.compareDiagnostics = &m_compareDiagnostics;
        ctx.abMetrics = &m_abMetrics;

        ctx.selectedBodyIndex = &m_selectedBodyIndex;
        ctx.entitySelectionJustChanged = &m_entitySelectionJustChanged;
        ctx.showFlowerOfLifeLatticeView = &m_showFlowerOfLifeLatticeView;
        ctx.fieldHeatmapMode = &m_fieldHeatmapMode;
        ctx.viewportAutoFit = &m_viewportAutoFit;
        ctx.viewportHalfExtent = &m_viewportHalfExtent;
        ctx.viewportPanX = &m_viewportPanX;
        ctx.viewportPanY = &m_viewportPanY;

        ctx.onRequestExit = [this]() {
            if (m_window != nullptr) {
                m_window->RequestClose();
            }
        };
        ctx.onPlay = [this]() {
            m_playbackStateRaw = static_cast<std::uint8_t>(ui::SimulationPlaybackState::Playing);
            m_simTimeAccumulator = 0.0;
            UpdateWindowTitle();
        };
        ctx.onPause = [this]() {
            m_playbackStateRaw = static_cast<std::uint8_t>(ui::SimulationPlaybackState::Paused);
            m_simTimeAccumulator = 0.0;
            UpdateWindowTitle();
        };
        ctx.onStop = [this]() {
            m_playbackStateRaw = static_cast<std::uint8_t>(ui::SimulationPlaybackState::Stopped);
            m_simTimeAccumulator = 0.0;
            if (m_hasActiveScene) {
                (void)ActivateSceneForSimulation(
                    m_activeScene,
                    m_activeScope,
                    m_primaryModelId,
                    m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                );
            } else {
                UpdateWindowTitle();
            }
        };
        ctx.onSingleStep = [this]() {
            m_playbackStateRaw = static_cast<std::uint8_t>(ui::SimulationPlaybackState::Paused);
            m_simTimeAccumulator = 0.0;
            if (StepActiveSolvers(1ULL, m_activeTimestep)) {
                RecordLiveTrails();
            }
            UpdateWindowTitle();
        };
        ctx.onResetSimulation = [this]() {
            if (m_hasActiveScene) {
                (void)ActivateSceneForSimulation(
                    m_activeScene,
                    m_activeScope,
                    m_primaryModelId,
                    m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                );
            }
        };

        ctx.onNewScene = [this]() { CreateNewEmptyScene(); };
        ctx.onOpenSceneFile = [this](const std::filesystem::path& path) { return OpenSceneFromPath(path); };
        ctx.onSaveSceneFile = [this](const std::filesystem::path& path) { return SaveActiveSceneToPath(path); };

        ctx.onActivateScene = [this](
            const scene::UniverseScene& scene,
            scene::SimulationScope scope,
            std::optional<scene::ModelId> explicitA,
            std::optional<scene::ModelId> explicitB
        ) {
            m_selectedBodyIndex.reset();
            return ActivateSceneForSimulation(scene, scope, explicitA, explicitB, false);
        };
        ctx.onSelectPrimaryModel = [this](scene::ModelId modelId) {
            if (m_hasActiveScene) {
                (void)ActivateSceneForSimulation(
                    m_activeScene,
                    m_activeScope,
                    modelId,
                    m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                );
            }
        };
        ctx.onSetCompareOverlay = [this](bool enabled, std::optional<scene::ModelId> modelB) {
            m_compareOverlayEnabled = enabled;
            if (modelB.has_value()) {
                m_compareModelId = modelB.value();
            }
            if (m_hasActiveScene) {
                (void)ActivateSceneForSimulation(
                    m_activeScene,
                    m_activeScope,
                    m_primaryModelId,
                    m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                );
            }
        };
        ctx.onSceneParametersChanged = [this]() {
            if (m_hasActiveScene) {
                (void)ActivateSceneForSimulation(
                    m_activeScene,
                    m_activeScope,
                    m_primaryModelId,
                    m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                );
            }
        };

        ctx.onCreateBodyEntity = [this](const scene::BodyEntity& entity) { AddBodyToActiveScene(entity); };
        ctx.onDeleteBodyEntity = [this](std::size_t idx) { DeleteBodyFromActiveScene(idx); };
        ctx.onFocusBodyEntity = [this](std::size_t idx) { FocusViewportOnBody(idx); };

        m_uiManager->Render(ctx);
    }
#endif

    int Application::Run() {
        if (!m_initialized) {
            spdlog::error("Application::Run() invoked before successful Initialize().");
            return 1;
        }
        if (m_options.profileOnly) {
            return 0;
        }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        if (m_guiActive && m_window != nullptr && m_renderDevice != nullptr) {
            auto prevFrameTime = std::chrono::steady_clock::now();
            m_simTimeAccumulator = 0.0;

            while (!m_window->ShouldClose()) {
                if (m_window->IsMinimized()) {
                    m_window->WaitEvents();
                    prevFrameTime = std::chrono::steady_clock::now();
                    m_simTimeAccumulator = 0.0;
                    continue;
                }
                m_window->PollEvents();

                const auto now = std::chrono::steady_clock::now();
                // Clamp wall-clock frame delta to <= 100 ms to avoid catch-up spikes
                // after window drags, OS stalls, or debugger pauses.
                const double frameDtSec = std::clamp(
                    std::chrono::duration<double>(now - prevFrameTime).count(),
                    0.0,
                    0.1
                );
                prevFrameTime = now;

                const bool isPlaying =
                    (m_playbackStateRaw == static_cast<std::uint8_t>(ui::SimulationPlaybackState::Playing));

                if (isPlaying && m_activeSolver != nullptr && m_activeSolver->IsInitialized() &&
                    m_activeScene.domain != scene::SimulationDomain::RadialAndScaleProfile) {
                    std::uint64_t subSteps = 0ULL;
                    if (m_wallClockPaced) {
                        const double dt = std::max(1.0e-12, m_activeTimestep);
                        m_simTimeAccumulator += frameDtSec * static_cast<double>(m_simTimeScale);
                        const std::uint64_t rawSteps = static_cast<std::uint64_t>(std::floor(m_simTimeAccumulator / dt));
                        const std::uint64_t maxCap = static_cast<std::uint64_t>(std::max(1, m_maxSubStepsPerFrame));
                        if (rawSteps > maxCap) {
                            subSteps = maxCap;
                            // Drop excess backlog beyond one step so the UI stays responsive.
                            m_simTimeAccumulator = std::min(m_simTimeAccumulator - (static_cast<double>(subSteps) * dt), dt);
                        } else {
                            subSteps = rawSteps;
                            m_simTimeAccumulator -= static_cast<double>(subSteps) * dt;
                        }
                    } else {
                        subSteps = static_cast<std::uint64_t>(std::max(1, m_stepsPerFrame));
                    }

                    m_lastFrameExecutedSteps = static_cast<int>(subSteps);
                    if (subSteps > 0ULL && StepActiveSolvers(subSteps, m_activeTimestep)) {
                        RecordLiveTrails();
                    }
                } else {
                    m_simTimeAccumulator = 0.0;
                    m_lastFrameExecutedSteps = 0;
                }

                if (m_renderDevice->BeginFrame()) {
                    RenderDesktopUi();
                    m_renderDevice->EndFrameAndPresent();
                }
            }
            return 0;
        }
#endif
        return RunHeadlessSimulation();
    }

    void Application::Shutdown() noexcept {
        if (!m_initialized && m_cudaContext == nullptr && m_activeSolver == nullptr && m_compareSolver == nullptr
#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
            && m_uiManager == nullptr && m_renderDevice == nullptr && m_window == nullptr
#endif
        ) {
            return;
        }

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
        // Shut down UIManager first so ImGuiConsoleSink detaches cleanly from
        // spdlog and panels are destroyed before ImGui/Vulkan teardown.
        if (m_uiManager != nullptr) {
            m_uiManager->Shutdown();
            m_uiManager.reset();
        }
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
