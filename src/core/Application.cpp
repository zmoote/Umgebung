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
    #include <imgui.h>
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

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        /** @brief Smooth perceptual plasma/inferno RGB ramp for 2D GPU field heatmaps (`t in [0, 1]`). */
        [[nodiscard]] ImU32 EvaluateHeatmapColor(float t, bool isDiffMap) noexcept {
            const float u = std::clamp(t, 0.0f, 1.0f);
            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (isDiffMap) {
                // Cool-indigo -> cyan -> bright white-gold for |A - B| difference maps.
                if (u < 0.5f) {
                    const float s = u * 2.0f;
                    r = 0.06f + (0.10f * s);
                    g = 0.08f + (0.68f * s);
                    b = 0.18f + (0.78f * s);
                } else {
                    const float s = (u - 0.5f) * 2.0f;
                    r = 0.16f + (0.84f * s);
                    g = 0.76f + (0.22f * s);
                    b = 0.96f - (0.35f * s);
                }
            } else {
                // Inferno ramp: deep navy -> magenta -> fiery orange -> bright yellow.
                if (u < 0.33f) {
                    const float s = u / 0.33f;
                    r = 0.05f + (0.50f * s);
                    g = 0.05f + (0.08f * s);
                    b = 0.15f + (0.45f * s);
                } else if (u < 0.67f) {
                    const float s = (u - 0.33f) / 0.34f;
                    r = 0.55f + (0.42f * s);
                    g = 0.13f + (0.37f * s);
                    b = 0.60f - (0.48f * s);
                } else {
                    const float s = (u - 0.67f) / 0.33f;
                    r = 0.97f + (0.03f * s);
                    g = 0.50f + (0.48f * s);
                    b = 0.12f + (0.68f * s);
                }
            }
            return IM_COL32(
                static_cast<int>(std::clamp(r, 0.0f, 1.0f) * 255.0f),
                static_cast<int>(std::clamp(g, 0.0f, 1.0f) * 255.0f),
                static_cast<int>(std::clamp(b, 0.0f, 1.0f) * 255.0f),
                255
            );
        }
#endif

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
        m_bodyTrails.assign(m_liveBodies.size(), {});
        m_compareTrails.assign(m_liveBodies.size(), {});
        m_fieldHeatmapMode = 0;
        m_simTimeAccumulator = 0.0;
        m_lastFrameExecutedSteps = 0;

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
            winConfig.preferX11OnLinux = m_options.forceX11;

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

    void Application::RenderSubSimulationsForParent(
        const scene::UniverseScene& sceneRef,
        const std::string& parentId
    ) {
        for (std::size_t sIdx = 0U; sIdx < sceneRef.subSimulations.size(); ++sIdx) {
            const auto& sub = sceneRef.subSimulations[sIdx];
            if (sub.parentId != parentId) {
                continue;
            }

            bool hasNestedSubs = false;
            for (const auto& cand : sceneRef.subSimulations) {
                if (cand.parentId == sub.id) {
                    hasNestedSubs = true;
                    break;
                }
            }

            const std::string nodeLabel =
                sub.name + " [" + scene::ScaleTierToString(sub.scaleTier) +
                " | " + scene::SimulationDomainToString(sub.domain) + "]##sub_" + sub.id;

            const ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_DefaultOpen |
                (hasNestedSubs ? ImGuiTreeNodeFlags_None : ImGuiTreeNodeFlags_Leaf);

            if (ImGui::TreeNodeEx(nodeLabel.c_str(), flags)) {
                ImGui::PushID(static_cast<int>(5000U + sIdx));
                if (ImGui::SmallButton("Zoom Into Sub-Simulation")) {
                    scene::UniverseScene subScene{};
                    std::string err;
                    if (scene::UniverseScene::LoadFromFile(sub.scenarioPath, subScene, err)) {
                        m_showFlowerOfLifeLatticeView = false;
                        (void)ActivateSceneForSimulation(subScene, scene::SimulationScope::Scenario);
                    } else {
                        spdlog::warn("Could not open sub-simulation '{}': {}", sub.scenarioPath.string(), err);
                    }
                }
                ImGui::PopID();

                if (hasNestedSubs) {
                    RenderSubSimulationsForParent(sceneRef, sub.id);
                }
                ImGui::TreePop();
            }
        }
    }

    void Application::RenderBodyHierarchyNode(
        const scene::UniverseScene& sceneRef,
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
            for (const auto& sub : sceneRef.subSimulations) {
                if (sub.parentId == body.id) {
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
                if (sceneRef.domain == scene::SimulationDomain::FieldGrid2D) {
                    ImGui::TextDisabled("Stationary | R=%.3f | pos=(%.2f, %.2f)", body.radius, body.position.x, body.position.y);
                } else {
                    ImGui::TextDisabled("m=%.6g | r=(%.3f, %.3f, %.3f)", body.mass, body.position.x, body.position.y, body.position.z);
                }
                if (hasChildren) {
                    RenderBodyHierarchyNode(sceneRef, bodies, body.id, showScaleTierTags);
                    RenderSubSimulationsForParent(sceneRef, body.id);
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

        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(8, 10, 22, 255));
        drawList->AddRect(canvasMin, canvasMax, IM_COL32(75, 65, 130, 255));

        const ImVec2 center(canvasMin.x + 0.5f * canvasSize.x, canvasMin.y + 0.5f * canvasSize.y);
        const float minDim = std::min(canvasSize.x, canvasSize.y);
        const double bubbleR = std::max(0.1, m_flowerMultiverse.GetBubbleRadius());
        const float pixelsPerUnit = (0.145f * minDim) / static_cast<float>(bubbleR);

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
                drawList->AddText(ImVec2(pos.x + 8.0f, pos.y - 8.0f), IM_COL32(240, 245, 255, 245), node.label.c_str());
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
                activeHalfExtent = std::max(activeHalfExtent, std::max(std::abs(b.position.x), std::abs(b.position.y)) * 1.25);
            }
            if (m_compareOverlayEnabled) {
                for (const auto& b : m_compareBodies) {
                    activeHalfExtent = std::max(activeHalfExtent, std::max(std::abs(b.position.x), std::abs(b.position.y)) * 1.25);
                }
            }
            m_viewportHalfExtent = static_cast<float>(activeHalfExtent);
        }

        ImGui::SameLine();
        ImGui::TextDisabled("| Domain: NBodyDynamics | +/-%.2f", activeHalfExtent);

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
            IM_COL32(255, 210, 90, 255), IM_COL32(95, 190, 255, 255),
            IM_COL32(255, 115, 145, 255), IM_COL32(130, 245, 165, 255),
            IM_COL32(195, 150, 255, 255), IM_COL32(255, 165, 85, 255)
        };

        if (m_compareOverlayEnabled) {
            for (std::size_t i = 0U; i < m_compareTrails.size(); ++i) {
                const auto& trail = m_compareTrails[i];
                if (trail.size() < 2U) { continue; }
                const ImU32 ghostColor = (kPalette[i % 6U] & 0x00FFFFFFU) | (150U << 24U);
                for (std::size_t k = 1U; k < trail.size(); k += 2U) {
                    const ImVec2 p0(center.x + static_cast<float>(trail[k - 1U].x) * pixelsPerUnit, center.y - static_cast<float>(trail[k - 1U].y) * pixelsPerUnit);
                    const ImVec2 p1(center.x + static_cast<float>(trail[k].x) * pixelsPerUnit, center.y - static_cast<float>(trail[k].y) * pixelsPerUnit);
                    drawList->AddLine(p0, p1, ghostColor, 1.3f);
                }
            }
        }

        for (std::size_t i = 0U; i < m_bodyTrails.size(); ++i) {
            const auto& trail = m_bodyTrails[i];
            if (trail.size() < 2U) { continue; }
            const ImU32 trailColor = (kPalette[i % 6U] & 0x00FFFFFFU) | (175U << 24U);
            for (std::size_t k = 1U; k < trail.size(); ++k) {
                const ImVec2 p0(center.x + static_cast<float>(trail[k - 1U].x) * pixelsPerUnit, center.y - static_cast<float>(trail[k - 1U].y) * pixelsPerUnit);
                const ImVec2 p1(center.x + static_cast<float>(trail[k].x) * pixelsPerUnit, center.y - static_cast<float>(trail[k].y) * pixelsPerUnit);
                drawList->AddLine(p0, p1, trailColor, 1.8f);
            }
        }

        for (std::size_t i = 0U; i < m_liveBodies.size(); ++i) {
            const auto& bA = m_liveBodies[i];
            const ImVec2 posA(center.x + static_cast<float>(bA.position.x) * pixelsPerUnit, center.y - static_cast<float>(bA.position.y) * pixelsPerUnit);
            const ImU32 color = kPalette[i % 6U];
            const float markerRadius = (bA.category == "Star" || bA.category == "GalacticCore") ? 7.0f : 5.0f;

            if (m_compareOverlayEnabled && i < m_compareBodies.size()) {
                const auto& bB = m_compareBodies[i];
                const ImVec2 posB(center.x + static_cast<float>(bB.position.x) * pixelsPerUnit, center.y - static_cast<float>(bB.position.y) * pixelsPerUnit);
                drawList->AddLine(posA, posB, IM_COL32(255, 255, 255, 85), 1.0f);
                drawList->AddCircle(posB, markerRadius + 2.5f, color, 24, 2.0f);
            }
            drawList->AddCircleFilled(posA, markerRadius, color);
            drawList->AddText(ImVec2(posA.x + markerRadius + 4.0f, posA.y - 7.0f), IM_COL32(225, 232, 245, 230), bA.name.c_str());
        }

        const std::string legendA = std::string("Model A (Solid): ") + scene::ModelIdToString(m_primaryModelId);
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 8.0f), IM_COL32(160, 230, 255, 240), legendA.c_str());
        if (m_compareOverlayEnabled) {
            const std::string legendB = std::string("Model B (Dashed/Ring): ") + scene::ModelIdToString(m_compareModelId);
            drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 24.0f), IM_COL32(255, 210, 130, 240), legendB.c_str());
        }
        drawList->PopClipRect();
    }

    void Application::RenderFieldGrid2DViewportCanvas() {
        ImGui::RadioButton("Model A Heatmap", &m_fieldHeatmapMode, 0);
        if (m_compareOverlayEnabled && !m_compareFieldSnap.scalarField.empty()) {
            ImGui::SameLine();
            ImGui::RadioButton("Model B Heatmap", &m_fieldHeatmapMode, 1);
            ImGui::SameLine();
            ImGui::RadioButton("|A - B| Difference", &m_fieldHeatmapMode, 2);
        } else {
            m_fieldHeatmapMode = 0;
        }

        const auto& snapA = m_primaryFieldSnap;
        const auto& snapB = m_compareFieldSnap;
        if (snapA.width == 0U || snapA.height == 0U || snapA.scalarField.empty()) {
            ImGui::TextDisabled("Waiting for 2D GPU FieldGrid2D snapshot...");
            return;
        }

        ImGui::TextDisabled(
            "Domain: FieldGrid2D (%ux%u) | %s | Peak=%.4g | CFL=%.3f",
            snapA.width, snapA.height, snapA.fieldLabel.c_str(), snapA.peakAbsVal, m_latestDiagnostics.effectiveCflNumber
        );

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float waveformHeight = 82.0f;
        const float mapHeight = std::max(120.0f, avail.y - waveformHeight - 12.0f);
        const float mapSide = std::min(avail.x, mapHeight);

        const ImVec2 mapMin = ImGui::GetCursorScreenPos();
        const ImVec2 mapMax(mapMin.x + mapSide, mapMin.y + mapSide);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(mapMin, mapMax, true);

        const bool isDiff = (m_fieldHeatmapMode == 2 && m_compareOverlayEnabled && snapB.scalarField.size() == snapA.scalarField.size());
        const bool showB = (m_fieldHeatmapMode == 1 && m_compareOverlayEnabled && snapB.scalarField.size() == snapA.scalarField.size());

        // Compute display range [lo, hi] cleanly.
        // For Thinker::TitorC204 (Model A), use a fixed physical metric distortion scale
        // [0.0, 3.5] (0 = flat Minkowski, 1.0 = Null Time Horizon, >1.0 = Negative CTC core)
        // so ramp-up expansion, 50/75/100% power levels, forward vs. backward energy differences,
        // VGL pulses, and arrival collapse are genuinely visible rather than normalized away!
        double lo = 0.0;
        double hi = 1.0e-12;
        if (isDiff) {
            hi = std::max(1.0e-12, m_abMetrics.fieldPeakResidual);
        } else if (!showB && m_primaryModelId == scene::ModelId::ThinkerTitorC204) {
            lo = 0.0;
            hi = 2.8;
        } else if (showB && m_compareModelId == scene::ModelId::ThinkerTitorC204) {
            lo = 0.0;
            hi = 2.8;
        } else {
            const auto& ref = showB ? snapB : snapA;
            lo = ref.minVal;
            hi = std::max(lo + 1.0e-12, ref.maxVal);
        }

        // Downsample stride to keep draw-list rectangles ultra-fast (~64x64 quads at 60+ FPS).
        const std::uint32_t stepX = std::max(1U, snapA.width / 64U);
        const std::uint32_t stepY = std::max(1U, snapA.height / 64U);
        const float cellPxW = mapSide / static_cast<float>((snapA.width + stepX - 1U) / stepX);
        const float cellPxH = mapSide / static_cast<float>((snapA.height + stepY - 1U) / stepY);

        std::uint32_t outY = 0U;
        for (std::uint32_t iy = 0U; iy < snapA.height; iy += stepY, ++outY) {
            std::uint32_t outX = 0U;
            for (std::uint32_t ix = 0U; ix < snapA.width; ix += stepX, ++outX) {
                const std::size_t idx = static_cast<std::size_t>(iy) * snapA.width + ix;
                double val = isDiff
                    ? std::abs(snapB.scalarField[idx] - snapA.scalarField[idx])
                    : (showB ? snapB.scalarField[idx] : snapA.scalarField[idx]);
                const float norm = static_cast<float>((val - lo) / (hi - lo));
                const ImU32 col = EvaluateHeatmapColor(norm, isDiff);
                const float x0 = mapMin.x + static_cast<float>(outX) * cellPxW;
                const float y0 = mapMax.y - static_cast<float>(outY + 1U) * cellPxH;
                drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + cellPxW + 0.5f, y0 + cellPxH + 0.5f), col);
            }
        }

        // Overlay stationary device geometry (HEEMFG ring, Vogel/Eredyon crystal + coils, OR C204 envelope + X-ray cone + singularities).
        const ImVec2 center(mapMin.x + 0.5f * mapSide, mapMin.y + 0.5f * mapSide);
        const float pxPerWorld = (0.5f * mapSide) / static_cast<float>(std::max(1.0e-6, snapA.domainHalfExtent));
        if (m_primaryModelId == scene::ModelId::ThinkerTitorC204) {
            const auto& tc = m_activeScene.physicsProfile.unifiedField.titorC204;
            // 1. 170-degree rearward X-ray radiation venting cone (Figure 2 Item 5).
            //    X-rays vent instantaneously at c (~27 ns across 8 m rear clearance), so
            //    the cone fill intensity tracks active field drive directly rather than drifting slowly.
            const float coneLenPx = static_cast<float>(tc.envelopeHalfLengthA * 0.95) * pxPerWorld;
            const float halfConeRad = static_cast<float>(0.5 * tc.xrayVentConeDeg * (3.14159265358979323846 / 180.0));
            const ImVec2 coneApex = center;
            const ImVec2 coneTop(coneApex.x - coneLenPx * std::cos(halfConeRad), coneApex.y - coneLenPx * std::sin(halfConeRad));
            const ImVec2 coneBot(coneApex.x - coneLenPx * std::cos(halfConeRad), coneApex.y + coneLenPx * std::sin(halfConeRad));
            const float xrayDrive = static_cast<float>(std::clamp(m_latestDiagnostics.c204CabinGForce / 1.85, 0.0, 1.0));
            if (xrayDrive > 0.02f) {
                const int fillAlpha = static_cast<int>(18.0f + 55.0f * xrayDrive);
                drawList->AddTriangleFilled(coneApex, coneTop, coneBot, IM_COL32(255, 70, 60, fillAlpha));
            }
            drawList->AddLine(coneApex, coneTop, IM_COL32(255, 95, 85, 200), 1.6f);
            drawList->AddLine(coneApex, coneBot, IM_COL32(255, 95, 85, 200), 1.6f);
            drawList->AddText(
                ImVec2(mapMin.x + 8.0f, center.y - 8.0f),
                IM_COL32(255, 140, 130, 235),
                (xrayDrive > 0.05f) ? "170 deg X-Ray Vent (@ c)" : "170 deg X-Ray Cone"
            );
            drawList->AddText(
                ImVec2(mapMin.x + 8.0f, mapMin.y + 6.0f),
                IM_COL32(190, 225, 255, 225),
                "Quasi-Stationary ADM Metric (GW & X-Ray Transit @ c ~ 37 ns / 11m)"
            );

            // 2. 22 m x 6 m Van Den Broeck ellipsoidal loading envelope centered at (x_offset, 0).
            const float envCx = center.x + static_cast<float>(tc.envelopeOffsetXM) * pxPerWorld;
            const float envCy = center.y;
            const float rx = static_cast<float>(tc.envelopeHalfLengthA) * pxPerWorld;
            const float ry = static_cast<float>(tc.envelopeHalfWidthB) * pxPerWorld;
            constexpr int kEllSegs = 64;
            for (int seg = 0; seg < kEllSegs; ++seg) {
                const float th0 = (2.0f * 3.14159265f * static_cast<float>(seg)) / static_cast<float>(kEllSegs);
                const float th1 = (2.0f * 3.14159265f * static_cast<float>(seg + 1)) / static_cast<float>(kEllSegs);
                const ImVec2 p0(envCx + rx * std::cos(th0), envCy - ry * std::sin(th0));
                const ImVec2 p1(envCx + rx * std::cos(th1), envCy - ry * std::sin(th1));
                drawList->AddLine(p0, p1, IM_COL32(130, 240, 255, 220), 2.0f);
            }
            drawList->AddText(ImVec2(envCx - 64.0f, envCy - ry - 16.0f), IM_COL32(150, 245, 255, 235), "22m x 6m Envelope");

            // 3. Live elliptical Lense-Thirring frame-dragging streamlines spanning the full 22m x 6m Kerr oval
            //    (retrograde CCW for backward CTC travel, prograde CW for forward Positive-Time travel,
            //    with differential outer angular speed reflecting Omega_drag(r) decay) + expanding VGL
            //    gravity-gradiometer scan oval during 240s VGL lock pulses.
            if (xrayDrive > 0.02f) {
                const float spinDir = m_latestDiagnostics.c204IsForwardTravel ? -1.0f : 1.0f;
                const float innerAngle = spinDir * static_cast<float>(m_latestDiagnostics.simulationTime * 2.8);
                const float outerAngle = spinDir * static_cast<float>(m_latestDiagnostics.simulationTime * 1.15);
                const float innerRx = 0.48f * rx;
                const float innerRy = 0.48f * ry;
                const float outerRx = 0.82f * rx;
                const float outerRy = 0.82f * ry;
                const ImU32 dragColInner = m_latestDiagnostics.c204IsForwardTravel
                    ? IM_COL32(110, 255, 185, static_cast<int>(195.0f * xrayDrive))
                    : IM_COL32(255, 225, 120, static_cast<int>(215.0f * xrayDrive));
                const ImU32 dragColOuter = m_latestDiagnostics.c204IsForwardTravel
                    ? IM_COL32(110, 255, 185, static_cast<int>(125.0f * xrayDrive))
                    : IM_COL32(255, 215, 120, static_cast<int>(135.0f * xrayDrive));
                for (int arc = 0; arc < 4; ++arc) {
                    const float offset = static_cast<float>(arc) * (2.0f * 3.14159265f / 4.0f);
                    const float a0 = innerAngle + offset;
                    const float b0 = outerAngle + offset + 0.4f;
                    constexpr float kArcSpan = 0.72f;
                    constexpr int kArcSteps = 10;
                    for (int s = 0; s < kArcSteps; ++s) {
                        const float f0 = static_cast<float>(s) / static_cast<float>(kArcSteps);
                        const float f1 = static_cast<float>(s + 1) / static_cast<float>(kArcSteps);
                        const float ta0 = a0 + kArcSpan * f0;
                        const float ta1 = a0 + kArcSpan * f1;
                        drawList->AddLine(
                            ImVec2(envCx + innerRx * std::cos(ta0), envCy - innerRy * std::sin(ta0)),
                            ImVec2(envCx + innerRx * std::cos(ta1), envCy - innerRy * std::sin(ta1)),
                            dragColInner, 1.8f
                        );
                        const float tb0 = b0 + kArcSpan * f0;
                        const float tb1 = b0 + kArcSpan * f1;
                        drawList->AddLine(
                            ImVec2(envCx + outerRx * std::cos(tb0), envCy - outerRy * std::sin(tb0)),
                            ImVec2(envCx + outerRx * std::cos(tb1), envCy - outerRy * std::sin(tb1)),
                            dragColOuter, 1.3f
                        );
                    }
                }
            }
            if (m_latestDiagnostics.c204OperationalState.find("VGL") != std::string::npos) {
                const float pulseFrac = static_cast<float>(std::fmod(m_latestDiagnostics.simulationTime * 0.65, 1.0));
                const float vglRx = (0.12f + 0.92f * pulseFrac) * rx;
                const float vglRy = (0.12f + 0.92f * pulseFrac) * ry;
                const int vglAlpha = static_cast<int>((1.0f - pulseFrac) * 230.0f);
                constexpr int kVglSegs = 48;
                for (int seg = 0; seg < kVglSegs; ++seg) {
                    const float th0 = (2.0f * 3.14159265f * static_cast<float>(seg)) / static_cast<float>(kVglSegs);
                    const float th1 = (2.0f * 3.14159265f * static_cast<float>(seg + 1)) / static_cast<float>(kVglSegs);
                    drawList->AddLine(
                        ImVec2(envCx + vglRx * std::cos(th0), envCy - vglRy * std::sin(th0)),
                        ImVec2(envCx + vglRx * std::cos(th1), envCy - vglRy * std::sin(th1)),
                        IM_COL32(110, 235, 255, vglAlpha), 2.0f
                    );
                }
            }
            drawList->AddText(
                ImVec2(mapMin.x + 8.0f, mapMax.y - 18.0f),
                IM_COL32(200, 220, 245, 215),
                "Fixed Scale |1+g_00|: 0.0 Minkowski | 1.0 Null Horizon | >1.0 Neg CTC"
            );
        }
        for (const auto& anchor : m_activeScene.bodies) {
            const ImVec2 p(
                center.x + static_cast<float>(anchor.position.x) * pxPerWorld,
                center.y - static_cast<float>(anchor.position.y) * pxPerWorld
            );
            const float rPx = static_cast<float>(anchor.radius) * pxPerWorld;
            if (anchor.category == "PhryllCrystal") {
                // Stationary double-terminated quartz crystal diamond silhouette.
                const float hw = rPx * 0.65f;
                const ImVec2 top(p.x, p.y - rPx * 1.15f);
                const ImVec2 bot(p.x, p.y + rPx * 0.85f);
                const ImVec2 lft(p.x - hw, p.y);
                const ImVec2 rgt(p.x + hw, p.y);
                drawList->AddQuad(top, rgt, bot, lft, IM_COL32(220, 250, 255, 220), 2.0f);
            } else if (anchor.category == "HEEMFG_Cavity") {
                drawList->AddCircle(p, rPx, IM_COL32(220, 250, 255, 210), 64, 2.0f);
            } else if (anchor.category == "C204_Envelope") {
                // Already rendered as the exact 22m x 6m Van Den Broeck ellipsoid above.
                continue;
            } else if (anchor.category == "C204_Singularity") {
                drawList->AddCircleFilled(p, std::max(4.0f, rPx), IM_COL32(255, 245, 140, 245));
                drawList->AddCircle(p, std::max(6.0f, rPx + 2.0f), IM_COL32(255, 160, 70, 230), 24, 1.5f);
            } else if (anchor.category == "C204_Cabin") {
                const float hw = std::max(8.0f, rPx * 1.4f);
                const float hh = std::max(5.0f, rPx * 0.9f);
                drawList->AddRect(ImVec2(p.x - hw, p.y - hh), ImVec2(p.x + hw, p.y + hh), IM_COL32(115, 255, 160, 235), 2.0f, 0, 2.0f);
                drawList->AddText(ImVec2(p.x + hw + 4.0f, p.y - 7.0f), IM_COL32(150, 255, 185, 235), "Cabin (1.8m)");
            } else {
                drawList->AddCircle(p, std::max(5.0f, rPx), IM_COL32(255, 230, 120, 220), 24, 1.8f);
            }
        }

        drawList->AddRect(mapMin, mapMax, IM_COL32(110, 145, 195, 255));
        drawList->PopClipRect();

        ImGui::Dummy(ImVec2(mapSide, mapSide + 4.0f));

        // Live center-probe time-series waveform strip underneath the 2D GPU heatmap!
        if (!m_probeWaveformA.empty()) {
            std::vector<float> waveA(m_probeWaveformA.begin(), m_probeWaveformA.end());
            const char* waveLabel = (m_primaryModelId == scene::ModelId::ThinkerTitorC204)
                ? "Cabin Metric g_00(t) [A]"
                : "Center Probe Ez(t) [A]";
            ImGui::PlotLines(waveLabel, waveA.data(), static_cast<int>(waveA.size()), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(avail.x * 0.72f, 58.0f));
        }
    }

    void Application::RenderRadialScaleViewportCanvas() {
        const auto& snapA = m_primaryRadialSnap;
        const auto& snapB = m_compareRadialSnap;
        if (snapA.xValues.size() < 2U) {
            ImGui::TextDisabled("No RadialAndScaleProfile data available.");
            return;
        }

        ImGui::TextDisabled(
            "Domain: RadialAndScaleProfile (%zu samples) | X: %s | Y: %s",
            snapA.xValues.size(), snapA.xLabel.c_str(), snapA.primaryYLabel.c_str()
        );

        const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
        const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        if (canvasSize.x <= 60.0f || canvasSize.y <= 60.0f) {
            return;
        }

        const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(canvasMin, canvasMax, true);

        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(10, 14, 24, 255));
        drawList->AddRect(canvasMin, canvasMax, IM_COL32(55, 80, 120, 255));

        const float padL = 52.0f, padR = 20.0f, padT = 36.0f, padB = 34.0f;
        const ImVec2 pMin(canvasMin.x + padL, canvasMin.y + padT);
        const ImVec2 pMax(canvasMax.x - padR, canvasMax.y - padB);
        const float plotW = std::max(10.0f, pMax.x - pMin.x);
        const float plotH = std::max(10.0f, pMax.y - pMin.y);

        auto mapX = [&](double x) -> double {
            return snapA.logarithmicX ? std::log10(std::max(x, 1.0e-300)) : x;
        };
        auto mapY = [&](double y) -> double {
            return snapA.logarithmicY ? std::log10(std::max(y, 1.0e-300)) : y;
        };

        double xMin = mapX(snapA.xValues.front());
        double xMax = mapX(snapA.xValues.back());
        if (xMax <= xMin) { xMax = xMin + 1.0; }

        double yMin = 1.0e300, yMax = -1.0e300;
        auto includeVec = [&](const std::vector<double>& v) {
            for (double val : v) {
                if (!std::isfinite(val)) { continue; }
                const double mv = mapY(val);
                if (std::isfinite(mv)) {
                    yMin = std::min(yMin, mv);
                    yMax = std::max(yMax, mv);
                }
            }
        };
        includeVec(snapA.primaryYValues);
        includeVec(snapA.secondaryYValues);
        if (m_compareOverlayEnabled) {
            includeVec(snapB.primaryYValues);
            includeVec(snapB.secondaryYValues);
        }
        if (yMax <= yMin) { yMin = 0.0; yMax = 1.0; }
        const double yPad = 0.06 * (yMax - yMin);
        yMin -= yPad;
        yMax += yPad;

        auto toScreen = [&](double x, double y) -> ImVec2 {
            const float u = static_cast<float>((mapX(x) - xMin) / (xMax - xMin));
            const float v = static_cast<float>((mapY(y) - yMin) / (yMax - yMin));
            return ImVec2(pMin.x + u * plotW, pMax.y - v * plotH);
        };

        // Gridlines
        for (int g = 0; g <= 4; ++g) {
            const float frac = static_cast<float>(g) / 4.0f;
            const float gx = pMin.x + frac * plotW;
            const float gy = pMax.y - frac * plotH;
            drawList->AddLine(ImVec2(gx, pMin.y), ImVec2(gx, pMax.y), IM_COL32(36, 50, 74, 150));
            drawList->AddLine(ImVec2(pMin.x, gy), ImVec2(pMax.x, gy), IM_COL32(36, 50, 74, 150));
        }

        // Physical landmark vertical markers
        auto drawLandmark = [&](double xVal, const char* label, ImU32 col) {
            const double mx = mapX(xVal);
            if (mx >= xMin && mx <= xMax) {
                const float sx = pMin.x + static_cast<float>((mx - xMin) / (xMax - xMin)) * plotW;
                drawList->AddLine(ImVec2(sx, pMin.y), ImVec2(sx, pMax.y), col, 1.2f);
                drawList->AddText(ImVec2(sx + 3.0f, pMin.y + 4.0f), col, label);
            }
        };
        if (m_activeScene.physicsProfile.radialScale.sweepKind == "HolographicMassScale") {
            drawLandmark(0.84184e-15, "Proton r_p", IM_COL32(150, 255, 175, 190));
            drawLandmark(2.5e4, "Cygnus X-1", IM_COL32(255, 220, 120, 190));
        } else if (m_activeScene.physicsProfile.radialScale.sweepKind == "C204MetricProfile") {
            const auto& tc = m_activeScene.physicsProfile.unifiedField.titorC204;
            drawLandmark(tc.cabinProbeDistanceM, "Cabin 1.8m", IM_COL32(150, 255, 175, 210));
            drawLandmark(tc.massOffsetM, "Mass Offset 5m", IM_COL32(255, 215, 120, 210));
            drawLandmark(tc.envelopeHalfLengthA + tc.envelopeOffsetXM, "Fwd Boundary", IM_COL32(130, 240, 255, 210));
        } else {
            const double Rs = m_activeScene.physicsProfile.radialScale.referenceSchwarzschildRadius;
            drawLandmark(Rs, "GR Rs", IM_COL32(255, 130, 130, 190));
            drawLandmark(1.5 * Rs, "R=1.5 Rs", IM_COL32(255, 215, 120, 190));
            drawLandmark(2.0 * Rs, "Essonne 2 Rs", IM_COL32(130, 240, 255, 190));
        }

        auto drawCurve = [&](const std::vector<double>& xs, const std::vector<double>& ys, ImU32 col, bool dashed, float thick) {
            const std::size_t n = std::min(xs.size(), ys.size());
            for (std::size_t k = 1U; k < n; k += (dashed ? 2U : 1U)) {
                drawList->AddLine(toScreen(xs[k - 1U], ys[k - 1U]), toScreen(xs[k], ys[k]), col, thick);
            }
        };

        drawCurve(snapA.xValues, snapA.primaryYValues, IM_COL32(95, 215, 255, 255), false, 2.2f);
        drawCurve(snapA.xValues, snapA.secondaryYValues, IM_COL32(255, 210, 90, 240), false, 1.7f);
        if (m_compareOverlayEnabled && !snapB.xValues.empty()) {
            drawCurve(snapB.xValues, snapB.primaryYValues, IM_COL32(255, 120, 145, 240), true, 2.0f);
            drawCurve(snapB.xValues, snapB.secondaryYValues, IM_COL32(180, 145, 255, 220), true, 1.5f);
        }

        drawList->AddRect(pMin, pMax, IM_COL32(95, 125, 175, 255));
        const std::string legA = std::string("A (Solid): ") + snapA.primaryYLabel + " / " + snapA.secondaryYLabel;
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 6.0f), IM_COL32(130, 225, 255, 245), legA.c_str());
        if (m_compareOverlayEnabled && !snapB.primaryYLabel.empty()) {
            const std::string legB = std::string("B (Dashed): ") + snapB.primaryYLabel + " / " + snapB.secondaryYLabel;
            drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 20.0f), IM_COL32(255, 150, 170, 245), legB.c_str());
        }
        drawList->PopClipRect();
    }

    void Application::RenderDesktopUi() {
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit", "Alt+F4")) { m_window->RequestClose(); }
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
            ImGui::Separator();
            ImGui::Text(
                "  %.1f FPS | Scope: %s | Domain: %s",
                io.Framerate,
                scene::SimulationScopeToString(m_activeScope),
                m_hasActiveScene ? scene::SimulationDomainToString(m_activeScene.domain) : "None"
            );
            ImGui::EndMainMenuBar();
        }

        // Hardware Panel
        if (m_showHardwarePanel) {
            ImGui::SetNextWindowSize(ImVec2(410.0f, 360.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Hardware & Adaptation", &m_showHardwarePanel)) {
                ImGui::Text("Engine Version : v%s (%s)", kEngineVersion, SystemProfiler::HardwareTierToString(m_adaptationPlan.tier));
                ImGui::Text("CPU            : %s (%u threads)", m_hardwareCaps.cpuModelName.c_str(), m_hardwareCaps.cpuLogicalCores);
                const double totalRamGiB = static_cast<double>(m_hardwareCaps.totalSystemRAMBytes) / kBytesPerGiB;
                const double availRamGiB = static_cast<double>(m_hardwareCaps.availableSystemRAMBytes) / kBytesPerGiB;
                ImGui::Text("Host RAM       : %.2f / %.2f GiB used", std::max(0.0, totalRamGiB - availRamGiB), totalRamGiB);
                ImGui::Text("Vulkan Device  : %s", m_renderDevice->GetSelectedVulkanGpuName().c_str());
                if (m_cudaContext != nullptr && m_cudaContext->IsInitialized()) {
                    ImGui::Text(
                        "CUDA Context   : GPU [%d] | %zu streams | VRAM: %s / %s",
                        m_cudaContext->GetDeviceIndex(),
                        m_cudaContext->GetStreamCount(),
                        compute::FormatBytes(m_cudaContext->GetTrackedAllocatedBytes()).c_str(),
                        compute::FormatBytes(m_cudaContext->GetVRAMBudgetBytes()).c_str()
                    );
                }
            }
            ImGui::End();
        }

        // Scope-Aware Workspace & Hierarchy Panel
        if (m_showHierarchyPanel) {
            ImGui::SetNextWindowSize(ImVec2(400.0f, 480.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Workspace & Hierarchy", &m_showHierarchyPanel)) {
                if (ImGui::BeginTabBar("SimulationScopeTabs")) {
                    if (ImGui::BeginTabItem("Scenarios")) {
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
                                sc.name + " [" + scene::SimulationDomainToString(sc.domain) + "]";
                            if (ImGui::TreeNodeEx(
                                    nodeLabel.c_str(),
                                    isSelected ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None
                                )) {
                                ImGui::TextDisabled("Default Model: %s", scene::ModelIdToString(sc.physicsProfile.defaultModel));
                                if (!sc.bodies.empty()) {
                                    RenderBodyHierarchyNode(sc, isSelected ? m_liveBodies : sc.bodies, "", false);
                                } else {
                                    ImGui::TextDisabled("Analytical 1D/Scale Sweep (%u samples)", sc.physicsProfile.radialScale.sampleCount);
                                }
                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                        }
                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Bubble Universes")) {
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
                                RenderBodyHierarchyNode(u, isSelected ? m_liveBodies : u.bodies, "", true);
                                RenderSubSimulationsForParent(u, "");
                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                        }
                        ImGui::EndTabItem();
                    }

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
                                if (!node.loadedUniverse.has_value()) { continue; }
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
                                ImGui::Text("[Ring %u] %s", node.ringIndex, node.label.c_str());
                                ImGui::PopID();
                            }
                        }
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
            ImGui::End();
        }

        // Domain-Filtered Simulation & A/B Comparison Panel
        if (m_showSimulationPanel) {
            ImGui::SetNextWindowSize(ImVec2(480.0f, 520.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Simulation & A/B Comparison", &m_showSimulationPanel)) {
                if (!m_hasActiveScene || m_activeSolver == nullptr || !m_activeSolver->IsInitialized()) {
                    ImGui::TextDisabled("Select a Scenario or Bubble Universe.");
                } else {
                    ImGui::Text("Active Scene  : %s", m_activeScene.name.c_str());
                    ImGui::TextColored(
                        ImVec4(0.55f, 0.90f, 0.70f, 1.0f),
                        "Domain        : %s (Models strictly filtered to compatible solvers)",
                        scene::SimulationDomainToString(m_activeScene.domain)
                    );

                    const auto compatibleModels = models::ModelRegistry::GetModelsForDomain(m_activeScene.domain);

                    const auto* curDescA = models::ModelRegistry::FindById(m_primaryModelId);
                    if (ImGui::BeginCombo("Primary Model A", curDescA ? curDescA->displayName : "Unknown")) {
                        for (const auto& desc : compatibleModels) {
                            const bool selected = (desc.id == m_primaryModelId);
                            const std::string itemLabel =
                                std::string("[") + models::ModelProvenanceToString(desc.provenance) + "] " + desc.displayName;
                            if (ImGui::Selectable(itemLabel.c_str(), selected) && desc.id != m_primaryModelId) {
                                (void)ActivateSceneForSimulation(
                                    m_activeScene, m_activeScope, desc.id,
                                    m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                                );
                            }
                            if (selected) { ImGui::SetItemDefaultFocus(); }
                        }
                        ImGui::EndCombo();
                    }

                    // Color-coded Epistemological Provenance Badge for Model A
                    if (curDescA != nullptr) {
                        ImVec4 badgeCol(0.55f, 0.90f, 0.65f, 1.0f);
                        if (curDescA->provenance == models::ModelProvenance::VerbatimEquation) {
                            badgeCol = ImVec4(0.45f, 0.85f, 1.0f, 1.0f);
                        } else if (curDescA->provenance == models::ModelProvenance::HybridVerbatimAndAnsatz) {
                            badgeCol = ImVec4(1.0f, 0.80f, 0.35f, 1.0f);
                        } else if (curDescA->provenance == models::ModelProvenance::OperationalizedSpec) {
                            badgeCol = ImVec4(0.85f, 0.60f, 1.0f, 1.0f);
                        }
                        ImGui::TextColored(badgeCol, "Provenance [A]: [%s]", models::ModelProvenanceToString(curDescA->provenance));
                        ImGui::TextWrapped("%s", curDescA->provenanceDetails);
                    }

                    bool overlayToggled = m_compareOverlayEnabled;
                    if (ImGui::Checkbox("Simultaneous Overlay (Model B)", &overlayToggled)) {
                        m_compareOverlayEnabled = overlayToggled;
                        (void)ActivateSceneForSimulation(
                            m_activeScene, m_activeScope, m_primaryModelId,
                            m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                        );
                    }

                    if (m_compareOverlayEnabled) {
                        const auto* curDescB = models::ModelRegistry::FindById(m_compareModelId);
                        if (ImGui::BeginCombo("Overlay Model B", curDescB ? curDescB->displayName : "Unknown")) {
                            for (const auto& desc : compatibleModels) {
                                const bool selected = (desc.id == m_compareModelId);
                                const std::string itemLabel =
                                    std::string("[") + models::ModelProvenanceToString(desc.provenance) + "] " + desc.displayName;
                                if (ImGui::Selectable(itemLabel.c_str(), selected) && desc.id != m_compareModelId) {
                                    (void)ActivateSceneForSimulation(
                                        m_activeScene, m_activeScope, m_primaryModelId, desc.id
                                    );
                                }
                                if (selected) { ImGui::SetItemDefaultFocus(); }
                            }
                            ImGui::EndCombo();
                        }
                        if (curDescB != nullptr) {
                            ImGui::TextDisabled("Provenance [B]: [%s]", models::ModelProvenanceToString(curDescB->provenance));
                        }
                    }

                    // Live tunable ansatz / hardware controls for immediate interactive A/B sensitivity testing.
                    if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D &&
                        m_primaryModelId == scene::ModelId::ThinkerVogelWillis) {
                        ImGui::Separator();
                        float driveV = static_cast<float>(m_activeScene.physicsProfile.unifiedField.vogelWillis.driveVoltageV);
                        if (ImGui::SliderFloat("Coil Drive Voltage (V)", &driveV, 5.0f, 80.0f, "%.1f V")) {
                            m_activeScene.physicsProfile.unifiedField.vogelWillis.driveVoltageV = static_cast<double>(driveV);
                            (void)ActivateSceneForSimulation(
                                m_activeScene, m_activeScope, m_primaryModelId,
                                m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                            );
                        }
                        const bool inSafe = (driveV >= 20.0f && driveV <= 60.0f);
                        ImGui::TextColored(
                            inSafe ? ImVec4(0.45f, 0.90f, 0.60f, 1.0f) : ImVec4(1.0f, 0.40f, 0.35f, 1.0f),
                            "  PhryllGen Spec Window [20-60 V]: %s",
                            inSafe ? "SAFE" : "OUT OF SPEC!"
                        );
                    } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D &&
                               m_primaryModelId == scene::ModelId::ThinkerPais) {
                        ImGui::Separator();
                        const double curKappa = m_activeScene.physicsProfile.unifiedField.pais.inertialReductionCoupling;
                        float logKappa = (curKappa > 0.0) ? static_cast<float>(std::log10(curKappa)) : -12.0f;
                        if (ImGui::SliderFloat("Pais Ansatz log10(kappa)", &logKappa, -13.0f, -8.0f, "10^%.2f")) {
                            m_activeScene.physicsProfile.unifiedField.pais.inertialReductionCoupling = std::pow(10.0, static_cast<double>(logKappa));
                            (void)ActivateSceneForSimulation(
                                m_activeScene, m_activeScope, m_primaryModelId,
                                m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                            );
                        }
                    } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D &&
                               m_primaryModelId == scene::ModelId::ThinkerTitorC204) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.50f, 0.90f, 1.0f, 1.0f), "GE C204 Pilot Controls (TMIM-200S/4005):");
                        auto& tc = m_activeScene.physicsProfile.unifiedField.titorC204;
                        bool c204Changed = false;
                        if (ImGui::SmallButton("50% (3.4 yr/hr)"))  { tc.powerLevelPercent = 50.0;  tc.injectedChargeCoulombs = 0.0; c204Changed = true; }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("75% (6.2 yr/hr)"))  { tc.powerLevelPercent = 75.0;  tc.injectedChargeCoulombs = 0.0; c204Changed = true; }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("100% (10.0 yr/hr)")) { tc.powerLevelPercent = 100.0; tc.injectedChargeCoulombs = 0.0; c204Changed = true; }

                        float pPct = static_cast<float>(tc.powerLevelPercent);
                        if (ImGui::SliderFloat("Matrix Power (%)", &pPct, 10.0f, 100.0f, "%.0f%%")) {
                            tc.powerLevelPercent = static_cast<double>(pPct);
                            tc.injectedChargeCoulombs = 0.0; // Auto-derive from manual power curve
                            c204Changed = true;
                        }
                        float targetYr = static_cast<float>(tc.targetDisplacementYears);
                        if (ImGui::SliderFloat("Target Jump (yrs)", &targetYr, -100.0f, 100.0f, "%+.1f yrs")) {
                            tc.targetDisplacementYears = static_cast<double>(targetYr);
                            c204Changed = true;
                        }
                        float payloadKg = static_cast<float>(tc.payloadMassKg);
                        if (ImGui::SliderFloat("Payload Mass (kg)", &payloadKg, 225.0f, 2800.0f, "%.0f kg")) {
                            tc.payloadMassKg = static_cast<double>(payloadKg);
                            c204Changed = true;
                        }
                        float vglFault = static_cast<float>(tc.simulatedVglAnomaly);
                        if (ImGui::SliderFloat("Inject VGL Gravity Anomaly", &vglFault, 0.0f, 0.30f, "%.3f")) {
                            tc.simulatedVglAnomaly = static_cast<double>(vglFault);
                            c204Changed = true;
                        }
                        if (c204Changed) {
                            (void)ActivateSceneForSimulation(
                                m_activeScene, m_activeScope, m_primaryModelId,
                                m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                            );
                        }
                    } else if (m_activeScene.domain == scene::SimulationDomain::NBodyDynamics &&
                               m_primaryModelId == scene::ModelId::ThinkerDanaan) {
                        ImGui::Separator();
                        float loveK = static_cast<float>(m_activeScene.physicsProfile.unifiedField.loveBindingCoupling);
                        if (ImGui::SliderFloat("Danaan Love Ansatz (kappa)", &loveK, 0.0f, 0.01f, "%.5f")) {
                            m_activeScene.physicsProfile.unifiedField.loveBindingCoupling = static_cast<double>(loveK);
                            (void)ActivateSceneForSimulation(
                                m_activeScene, m_activeScope, m_primaryModelId,
                                m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                            );
                        }
                    }

                    ImGui::Separator();

                    if (m_activeScene.domain != scene::SimulationDomain::RadialAndScaleProfile) {
                        if (ImGui::Button(m_simulationRunning ? "Pause" : "Play")) {
                            m_simulationRunning = !m_simulationRunning;
                            m_simTimeAccumulator = 0.0;
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Single Step")) {
                            m_simulationRunning = false;
                            m_simTimeAccumulator = 0.0;
                            if (StepActiveSolvers(1ULL, m_activeTimestep)) { RecordLiveTrails(); }
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Reset")) {
                            (void)ActivateSceneForSimulation(
                                m_activeScene, m_activeScope, m_primaryModelId,
                                m_compareOverlayEnabled ? std::optional<scene::ModelId>(m_compareModelId) : std::nullopt
                            );
                        }

                        if (ImGui::Checkbox("Wall-Clock Paced (FPS-Independent)", &m_wallClockPaced)) {
                            m_simTimeAccumulator = 0.0;
                        }

                        if (m_wallClockPaced) {
                            ImGui::SliderFloat(
                                "Time Scale",
                                &m_simTimeScale,
                                0.05f,
                                600.0f,
                                "%.2fx (sim-t / real-s)",
                                ImGuiSliderFlags_Logarithmic
                            );
                            if (ImGui::SmallButton("0.25x")) { m_simTimeScale = 0.25f; }
                            ImGui::SameLine();
                            if (ImGui::SmallButton("1x"))    { m_simTimeScale = 1.0f; }
                            ImGui::SameLine();
                            if (ImGui::SmallButton("10x"))   { m_simTimeScale = 10.0f; }
                            ImGui::SameLine();
                            if (ImGui::SmallButton("60x"))   { m_simTimeScale = 60.0f; }
                            ImGui::SameLine();
                            if (ImGui::SmallButton("300x"))  { m_simTimeScale = 300.0f; }

                            ImGui::TextDisabled(
                                "Fixed dt=%.4g | Last Frame Steps=%d%s",
                                m_activeTimestep,
                                m_lastFrameExecutedSteps,
                                (m_lastFrameExecutedSteps >= m_maxSubStepsPerFrame) ? " [CAPPED]" : ""
                            );
                        } else {
                            ImGui::SliderInt("Steps / Frame", &m_stepsPerFrame, 1, 60);
                        }
                    }

                    // Domain-adaptive primary telemetry readout
                    ImGui::Separator();
                    if (m_activeScene.domain == scene::SimulationDomain::NBodyDynamics) {
                        ImGui::Text("Step: %llu | t=%.5f | E=%.8f (drift=%.2e)",
                            static_cast<unsigned long long>(m_latestDiagnostics.stepCount),
                            m_latestDiagnostics.simulationTime,
                            m_latestDiagnostics.totalEnergy,
                            m_latestDiagnostics.relativeEnergyDrift
                        );
                        ImGui::Text("|P|=%.2e | |L|=%.8f", m_latestDiagnostics.totalLinearMomentum.Norm(), m_latestDiagnostics.totalAngularMomentum.Norm());
                    } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D) {
                        ImGui::Text("Step: %llu | t=%.4f | CFL=%.3f | Field Energy=%.5g",
                            static_cast<unsigned long long>(m_latestDiagnostics.stepCount),
                            m_latestDiagnostics.simulationTime,
                            m_latestDiagnostics.effectiveCflNumber,
                            m_latestDiagnostics.fieldEnergy
                        );
                        ImGui::Text("Field L2: %.5g | Peak: %.5g | Center Probe: %.5g",
                            m_latestDiagnostics.fieldL2Norm,
                            m_latestDiagnostics.fieldPeakValue,
                            m_latestDiagnostics.fieldProbeValue
                        );
                    } else {
                        ImGui::Text("Analytical Sweep Samples: %zu", m_primaryRadialSnap.xValues.size());
                        ImGui::Text("<P/P0>: %.6f | <c'/c>: %.6f | <Haramein phi>: %.3e",
                            m_latestDiagnostics.meanVacuumPressureRatio,
                            m_latestDiagnostics.meanLocalLightSpeedRatio,
                            m_latestDiagnostics.meanHolographicPhi
                        );
                    }

                    if (m_latestDiagnostics.hasUnifiedTelemetry) {
                        ImGui::Separator();
                        ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "Thinker Telemetry (Model A):");
                        if (m_latestDiagnostics.paisMaxPoyntingFlux > 0.0) {
                            ImGui::Text("  Pais S_max     : %.4e W/m^2 | m_eff/m0: %.6f",
                                m_latestDiagnostics.paisMaxPoyntingFlux, m_latestDiagnostics.paisInertialMassFactor);
                        }
                        if (m_latestDiagnostics.totalPhryllFlux > 0.0 || m_latestDiagnostics.vogelCrystalSaturationRatio > 0.0) {
                            ImGui::Text("  Vogel Phryll   : %.4g | Saturation Ratio: %.4f",
                                m_latestDiagnostics.totalPhryllFlux, m_latestDiagnostics.vogelCrystalSaturationRatio);
                        }
                        if (m_latestDiagnostics.vogelCrystalSinging) {
                            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "  WARNING: CRYSTAL SATURATION / SINGING THRESHOLD EXCEEDED!");
                        }
                        if (m_latestDiagnostics.hasC204Telemetry) {
                            ImGui::Separator();
                            ImGui::TextColored(ImVec4(0.55f, 0.92f, 1.0f, 1.0f), "C204 Flight Computer & Effective Metric Telemetry:");
                            ImGui::Text("  State         : %s", m_latestDiagnostics.c204OperationalState.c_str());
                            ImGui::TextColored(
                                m_latestDiagnostics.c204IsForwardTravel
                                    ? ImVec4(0.45f, 0.95f, 0.65f, 1.0f)
                                    : ImVec4(1.0f, 0.72f, 0.40f, 1.0f),
                                "  Direction     : %s",
                                m_latestDiagnostics.c204IsForwardTravel
                                    ? "FORWARD (+Time Dilation, -25% Energy)"
                                    : "BACKWARD (Retrograde CTC, 100% Energy)"
                            );
                            ImGui::Text("  Worldline     : Year %.3f (dT=%+.3f yr) | Div: %.3f%%",
                                m_latestDiagnostics.c204CurrentWorldlineYear,
                                m_latestDiagnostics.c204ElapsedDisplacedYears,
                                m_latestDiagnostics.c204WorldlineDivergencePct);
                            ImGui::Text("  Drive Matrix  : %.0f%% | %.0f C (Used: %.0f C) | %.2f yrs/hr (v_t=%.0f s/s)",
                                m_latestDiagnostics.c204PowerLevelPercent,
                                m_latestDiagnostics.c204InjectedChargeCoulombs,
                                m_latestDiagnostics.c204ExpendedChargeCoulombs,
                                m_latestDiagnostics.c204PenetrationRateYrsPerHr,
                                m_latestDiagnostics.c204TimeVelocitySecPerSec);
                            ImGui::Text("  Cabin Metric  : g_00=%.4f | %.2f G | |grad g_00|=%.2e m^-1",
                                m_latestDiagnostics.c204MetricG00AtCabin,
                                m_latestDiagnostics.c204CabinGForce,
                                m_latestDiagnostics.c204MetricGradG00AtCabin);
                            ImGui::Text("  Environment   : Hull %.1f F | O2 Pocket %.1f%% | %s",
                                m_latestDiagnostics.c204ExteriorTempF,
                                m_latestDiagnostics.c204CabinOxygenPct,
                                m_latestDiagnostics.c204OpticalPhase.c_str());
                            ImGui::Text("  VGL / Borda   : Checks=%llu | Next Pulse=%.0fs | Spread=%.2e",
                                static_cast<unsigned long long>(m_latestDiagnostics.c204VglCheckCount),
                                m_latestDiagnostics.c204VglNextCheckCountdownSec,
                                m_latestDiagnostics.c204BordaConsensusSpread);
                            ImGui::Text("  Horizons      : Neg=%llu | Null=%llu | Pos=%llu",
                                static_cast<unsigned long long>(m_latestDiagnostics.c204NegativeHorizonCells),
                                static_cast<unsigned long long>(m_latestDiagnostics.c204NullHorizonCells),
                                static_cast<unsigned long long>(m_latestDiagnostics.c204PositiveHorizonCells));
                            ImGui::Text("  Kerr Photons  : Prograde r1=%.2f M | Retrograde r2=%.2f M",
                                m_latestDiagnostics.c204KerrProgradePhotonR,
                                m_latestDiagnostics.c204KerrRetrogradePhotonR);
                            if (m_latestDiagnostics.c204UvBurstActive) {
                                ImGui::TextColored(ImVec4(0.85f, 0.55f, 1.0f, 1.0f), "  CAUTION: 1s UV RADIATION BURST - WEAR EYE PROTECTION!");
                            }
                            if (m_latestDiagnostics.c204CarbonSensorAlert) {
                                ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f), "  WARNING: CABIN CARBON SENSOR ALERT - ENGAGE REBREATHER!");
                            }
                            if (m_latestDiagnostics.c204OriginReturnTriggered) {
                                ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "  ABORT: %s", m_latestDiagnostics.c204AbortReason.c_str());
                            }
                        }
                    }

                    // Domain-adaptive A/B Divergence readout
                    if (m_abMetrics.active) {
                        ImGui::Separator();
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.82f, 0.4f, 1.0f),
                            "Live A/B Comparison: %s vs %s",
                            scene::ModelIdToString(m_primaryModelId),
                            scene::ModelIdToString(m_compareModelId)
                        );
                        if (m_activeScene.domain == scene::SimulationDomain::NBodyDynamics) {
                            ImGui::Text("Max dR: %.4e | RMS dR: %.4e | Max dV: %.4e | Max dTau: %.4e",
                                m_abMetrics.maxDeltaR, m_abMetrics.rmsDeltaR, m_abMetrics.maxDeltaV, m_abMetrics.maxDeltaProperTime);
                        } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D) {
                            ImGui::Text("Grid L2 Residual: %.4e | Peak |A-B|: %.4e", m_abMetrics.fieldL2Residual, m_abMetrics.fieldPeakResidual);
                            ImGui::Text("Peak (A): %.4e | Peak (B): %.4e", m_abMetrics.fieldPeakA, m_abMetrics.fieldPeakB);
                        } else {
                            ImGui::Text("Curve Max |A-B|: %.4e | RMS: %.4e | Max Rel Diff: %.4e",
                                m_abMetrics.curveMaxAbsDiff, m_abMetrics.curveRmsDiff, m_abMetrics.curveMaxRelDiff);
                        }
                    }
                }
            }
            ImGui::End();
        }

        // Domain-Adaptive 2D Viewport
        if (m_showViewport2DPanel) {
            ImGui::SetNextWindowSize(ImVec2(580.0f, 580.0f), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("2D Simulation Viewport", &m_showViewport2DPanel)) {
                if (m_showFlowerOfLifeLatticeView && !m_flowerMultiverse.IsEmpty()) {
                    RenderFlowerOfLifeViewportCanvas();
                } else if (!m_hasActiveScene) {
                    ImGui::TextDisabled("No active scene loaded.");
                } else if (m_activeScene.domain == scene::SimulationDomain::FieldGrid2D) {
                    RenderFieldGrid2DViewportCanvas();
                } else if (m_activeScene.domain == scene::SimulationDomain::RadialAndScaleProfile) {
                    RenderRadialScaleViewportCanvas();
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

                if (m_simulationRunning && m_activeSolver != nullptr && m_activeSolver->IsInitialized() &&
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
            && m_renderDevice == nullptr && m_window == nullptr
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
