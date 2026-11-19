/**
 * @file Application.hpp
 * @brief Top-level lifecycle orchestrator for the Umgebung Reality Simulation Engine.
 *
 * @details
 * `umgebung::core::Application` coordinates all engine pillars across both
 * Desktop Interactive GUI builds (`UMGEBUNG_HAS_GUI == 1`) and Headless HPC
 * Batch builds (`UMGEBUNG_HAS_GUI == 0`):
 *   1. **CLI Parsing (`cxxopts`):** Supports hardware profiling (`--profile-only`,
 *      `--json-profile`, `--device`), scene/scope loading (`--scene`), composable
 *      three-tier Model Registry selection (`--model`, `--compare-model`), batch
 *      stepping (`--steps`, `--dt`, `--output`), and desktop GUI options
 *      (`--headless`, `--width`, `--height`, `--x11`, `--log-level`).
 *   2. **Startup Hardware Discovery & Adaptation (`SystemProfiler`):** Probes
 *      CPU, host RAM, and every NVIDIA GPU, deriving a `HardwareAdaptationPlan`.
 *   3. **GPU Compute Initialization (`compute::CUDAContext`):** Binds the
 *      adapted primary GPU, creates non-blocking CUDA streams (stream 0 for
 *      Primary Model A; stream 1 for simultaneous Comparison Overlay Model B),
 *      and enforces the adapted VRAM budget.
 *   4. **Three Explicit Simulation Scopes (`scene/`):**
 *        - **Scenarios** (`scenes/scenarios/<id>.scenario.umgebung`),
 *        - **Bubble Universes** (`scenes/universes/<id>.umgebung`),
 *        - **Flower of Life Multiverse** (`scenes/multiverses/<id>.multiverse.umgebung`).
 *   5. **Composable Three-Tier Model Registry & Simultaneous GPU A/B Overlay:**
 *      Instantiates Primary Model A (`m_activeSolver`) and optional Comparison
 *      Overlay Model B (`m_compareSolver`) from `models::ModelRegistry`,
 *      stepping both identically on the GPU and computing live quantitative
 *      divergence ($\Delta r, \Delta v, \Delta E, \Delta \tau$).
 *   6. **Conditional Desktop GUI (`Window` + `RenderDevice`, GUI builds only):**
 *      Scope-aware Workspace & Hierarchy Inspector, Simulation & A/B Comparison
 *      panel, Hardware Telemetry, and 2D Orbital / Flower of Life Viewport.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/FlowerOfLifeMultiverse.hpp"
#include "umgebung/scene/Multiverse.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
namespace umgebung::graphics {
    class RenderDevice;
    class Window;
}
#endif

namespace umgebung::core {

    /**
     * @enum ApplicationInitResult
     * @brief Tri-state outcome returned by `Application::Initialize()`.
     */
    enum class ApplicationInitResult : std::uint8_t {
        /** @brief All startup stages succeeded; caller should invoke `Run()`. */
        Ready = 0,
        /** @brief Handled an informational flag (`--help` / `--version`); exit with `0`. */
        EarlyExitSuccess = 1,
        /** @brief Startup failed (invalid CLI, missing CUDA GPU, scene/solver/window failure). */
        Failure = 2
    };

    /**
     * @struct BodyDivergenceRecord
     * @brief Per-entity quantitative divergence between Primary Model A and
     *        simultaneous Comparison Overlay Model B at the same coordinate time $t$.
     */
    struct BodyDivergenceRecord {
        /** @brief Entity display name copied from the active scene. */
        std::string name;

        /** @brief Euclidean position separation $\Delta r_i = \|\mathbf{r}_i^{(B)} - \mathbf{r}_i^{(A)}\|$. */
        double deltaR{0.0};

        /** @brief Euclidean velocity separation $\Delta v_i = \|\mathbf{v}_i^{(B)} - \mathbf{v}_i^{(A)}\|$. */
        double deltaV{0.0};

        /** @brief Local proper-time divergence $\Delta \tau_i = |\tau_i^{(B)} - \tau_i^{(A)}|$. */
        double deltaProperTime{0.0};
    };

    /**
     * @struct ABComparisonMetrics
     * @brief Aggregate and per-body divergence telemetry produced when
     *        Simultaneous GPU Overlay Comparison (Model A vs. Model B) is active.
     */
    struct ABComparisonMetrics {
        /** @brief True when Model B is initialized and actively stepped alongside Model A. */
        bool active{false};

        /** @brief Maximum Euclidean position divergence across all bodies $\max_i \Delta r_i$. */
        double maxDeltaR{0.0};

        /** @brief Root-mean-square position divergence $\sqrt{\frac{1}{N}\sum_i \Delta r_i^2}$. */
        double rmsDeltaR{0.0};

        /** @brief Maximum velocity divergence $\max_i \Delta v_i$. */
        double maxDeltaV{0.0};

        /** @brief Signed total mechanical energy difference $\Delta E = E_B - E_A$. */
        double deltaTotalEnergy{0.0};

        /** @brief Maximum local proper-time divergence $\max_i |\tau_i^{(B)} - \tau_i^{(A)}|$. */
        double maxDeltaProperTime{0.0};

        /** @brief Per-body breakdown in scene index order. */
        std::vector<BodyDivergenceRecord> perBody;
    };

    /**
     * @struct ApplicationOptions
     * @brief Parsed command-line configuration governing a single `Application` run.
     */
    struct ApplicationOptions {
        bool showHelp{false};
        bool showVersion{false};
        bool forceHeadless{false};
        bool profileOnly{false};
        bool dumpHardwareJson{false};

        /** @brief Optional CUDA GPU device index requested via `--device <id>`. */
        std::optional<int> preferredGpuDevice{std::nullopt};

        /** @brief Optional path to a `.scenario.umgebung`, `.umgebung`, `.multiverse.umgebung`, or directory (`--scene`). */
        std::optional<std::filesystem::path> scenePath{std::nullopt};

        /** @brief Optional Primary Model A override from `ModelRegistry` (`--model <name>`). */
        std::optional<scene::ModelId> overridePrimaryModel{std::nullopt};

        /** @brief Optional Simultaneous Overlay Model B from `ModelRegistry` (`--compare-model <name>`). */
        std::optional<scene::ModelId> overrideCompareModel{std::nullopt};

        /** @brief Optional headless batch step count override (`--steps`). */
        std::optional<std::uint64_t> overrideSteps{std::nullopt};

        /** @brief Optional timestep $\Delta t$ override (`--dt`). */
        std::optional<double> overrideTimestep{std::nullopt};

        /** @brief Optional JSON telemetry output path for headless batch runs (`--output`). */
        std::optional<std::filesystem::path> outputPath{std::nullopt};

        std::uint32_t windowWidth{1280};
        std::uint32_t windowHeight{720};
        bool forceX11{false};
        std::string logLevel{"info"};
    };

    /**
     * @class Application
     * @brief Owns and coordinates the engine startup sequence, hardware
     *        adaptation state, CUDA compute context, three-scope scene workspace,
     *        simultaneous Model A / Model B GPU solvers, and optional Desktop GUI.
     */
    class Application final {
    public:
        static constexpr const char* kEngineVersion = "0.0.1";

        Application();
        ~Application();

        Application(const Application&) = delete;
        Application& operator=(const Application&) = delete;
        Application(Application&&) = delete;
        Application& operator=(Application&&) = delete;

        [[nodiscard]] ApplicationInitResult Initialize(int argc, char** argv);
        [[nodiscard]] int Run();
        void Shutdown() noexcept;

        [[nodiscard]] const ApplicationOptions& GetOptions() const noexcept { return m_options; }
        [[nodiscard]] const HardwareCapabilities& GetHardwareCapabilities() const noexcept { return m_hardwareCaps; }
        [[nodiscard]] const HardwareAdaptationPlan& GetAdaptationPlan() const noexcept { return m_adaptationPlan; }
        [[nodiscard]] compute::CUDAContext* GetCudaContext() const noexcept { return m_cudaContext.get(); }
        [[nodiscard]] const scene::Multiverse& GetScenarios() const noexcept { return m_scenarios; }
        [[nodiscard]] const scene::Multiverse& GetUniverses() const noexcept { return m_universes; }
        [[nodiscard]] const scene::FlowerOfLifeMultiverse& GetFlowerOfLifeMultiverse() const noexcept {
            return m_flowerMultiverse;
        }
        [[nodiscard]] const ABComparisonMetrics& GetABComparisonMetrics() const noexcept { return m_abMetrics; }
        [[nodiscard]] bool IsGuiActive() const noexcept { return m_guiActive; }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        [[nodiscard]] graphics::Window* GetWindow() const noexcept { return m_window.get(); }
        [[nodiscard]] graphics::RenderDevice* GetRenderDevice() const noexcept { return m_renderDevice.get(); }
#endif

    private:
        [[nodiscard]] ApplicationInitResult ParseCommandLine(int argc, char** argv);
        void ConfigureLogging() const;
        [[nodiscard]] bool LoadInitialWorkspace(const char* argv0);
        [[nodiscard]] bool ActivateSceneForSimulation(
            const scene::UniverseScene& scene,
            scene::SimulationScope activeScope,
            std::optional<scene::ModelId> explicitPrimaryModel = std::nullopt,
            std::optional<scene::ModelId> explicitCompareModel = std::nullopt
        );
        [[nodiscard]] bool StepActiveSolvers(std::uint64_t stepCount, double dt);
        void UpdateABComparisonMetrics();
        [[nodiscard]] int RunHeadlessSimulation();

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        void RenderDesktopUi();
        void RenderBodyHierarchyNode(
            const std::vector<scene::BodyEntity>& bodies,
            const std::string& parentId,
            bool showScaleTierTags
        );
        void RenderFlowerOfLifeViewportCanvas();
        void RenderOrbitalViewportCanvas();
        void RecordLiveTrails();
#endif

        ApplicationOptions m_options{};
        HardwareCapabilities m_hardwareCaps{};
        HardwareAdaptationPlan m_adaptationPlan{};
        std::unique_ptr<compute::CUDAContext> m_cudaContext;

        // Three Explicit Simulation Scopes
        scene::Multiverse m_scenarios{};
        scene::Multiverse m_universes{};
        scene::FlowerOfLifeMultiverse m_flowerMultiverse{};

        scene::SimulationScope m_activeScope{scene::SimulationScope::Scenario};
        bool m_hasActiveScene{false};
        scene::UniverseScene m_activeScene{};

        // Primary Model A & Simultaneous Comparison Overlay Model B
        scene::ModelId m_primaryModelId{scene::ModelId::MainstreamNewtonian};
        bool m_compareOverlayEnabled{false};
        scene::ModelId m_compareModelId{scene::ModelId::ThinkerEssonne};

        std::unique_ptr<models::IPhysicsSolver> m_activeSolver;
        std::unique_ptr<models::IPhysicsSolver> m_compareSolver;

        std::vector<scene::BodyEntity> m_liveBodies;
        std::vector<scene::BodyEntity> m_compareBodies;
        std::vector<double> m_primaryProperTimes;
        std::vector<double> m_compareProperTimes;

        models::SolverDiagnostics m_latestDiagnostics{};
        models::SolverDiagnostics m_compareDiagnostics{};
        ABComparisonMetrics m_abMetrics{};

        double m_activeTimestep{1.0e-3};
        bool m_initialized{false};
        bool m_guiActive{false};

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        std::unique_ptr<graphics::Window> m_window;
        std::unique_ptr<graphics::RenderDevice> m_renderDevice;

        bool m_simulationRunning{true};
        int m_stepsPerFrame{10};

        bool m_showHardwarePanel{true};
        bool m_showHierarchyPanel{true};
        bool m_showSimulationPanel{true};
        bool m_showViewport2DPanel{true};
        bool m_showFlowerOfLifeLatticeView{false};

        bool m_viewportAutoFit{false};
        float m_viewportHalfExtent{6.5f};

        std::vector<std::deque<scene::Vec3d>> m_bodyTrails;
        std::vector<std::deque<scene::Vec3d>> m_compareTrails;
        static constexpr std::size_t kMaxTrailPoints = 400U;
#endif
    };

} // namespace umgebung::core
