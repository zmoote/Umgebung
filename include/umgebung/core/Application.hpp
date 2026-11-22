/**
 * @file Application.hpp
 * @brief Top-level lifecycle orchestrator for the Umgebung Reality Simulation Engine.
 *
 * @details
 * Coordinates all engine pillars across Desktop Interactive GUI builds
 * (`UMGEBUNG_HAS_GUI == 1`) and Headless HPC Batch builds (`UMGEBUNG_HAS_GUI == 0`):
 *   1. **CLI Parsing (`cxxopts`):** Hardware profiling (`--profile-only`,
 *      `--json-profile`, `--device`), scene/scope loading (`--scene`),
 *      domain-filtered three-tier Model Registry selection (`--model`,
 *      `--compare-model`), batch stepping (`--steps`, `--dt`, `--output`), and
 *      desktop GUI options (`--headless`, `--width`, `--height`, `--x11`, `--log-level`).
 *   2. **Startup Hardware Discovery & Adaptation (`SystemProfiler`).**
 *   3. **GPU Compute Initialization (`compute::CUDAContext`).**
 *   4. **Three Explicit Simulation Scopes (`scene/`):**
 *        - Scenarios (`scenes/scenarios/<id>.scenario.umgebung`),
 *        - Bubble Universes (`scenes/universes/<id>.umgebung`) with multi-scale
 *          domain-appropriate `sub_simulations`,
 *        - Flower of Life Multiverse (`scenes/multiverses/<id>.multiverse.umgebung`).
 *   5. **Explicit `SimulationDomain` Dispatch & Simultaneous GPU A/B Overlay:**
 *        - `NBodyDynamics`         -> Orbital particle SoA solvers + ghost trajectory overlay
 *        - `FieldGrid2D`           -> GPU 2D TMz FDTD / Cavity / Torsion / Vacuum heatmap + A-B difference map + waveform
 *        - `RadialAndScaleProfile` -> 1D radial / logarithmic multi-scale curve & residual plotter
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
namespace umgebung::ui {
    class UIManager;
    enum class SimulationPlaybackState : std::uint8_t;
}
#endif

namespace umgebung::core {

    enum class ApplicationInitResult : std::uint8_t {
        Ready = 0,
        EarlyExitSuccess = 1,
        Failure = 2
    };

    /**
     * @struct BodyDivergenceRecord
     * @brief Per-entity quantitative divergence for `NBodyDynamics` A/B overlays.
     */
    struct BodyDivergenceRecord {
        std::string name;
        double deltaR{0.0};
        double deltaV{0.0};
        double deltaProperTime{0.0};
    };

    /**
     * @struct ABComparisonMetrics
     * @brief Domain-adaptive aggregate & per-entity divergence telemetry produced
     *        when Simultaneous Model A + Model B Overlay Comparison is active.
     */
    struct ABComparisonMetrics {
        bool active{false};
        scene::SimulationDomain domain{scene::SimulationDomain::NBodyDynamics};

        // NBodyDynamics metrics
        double maxDeltaR{0.0};
        double rmsDeltaR{0.0};
        double maxDeltaV{0.0};
        double deltaTotalEnergy{0.0};
        double maxDeltaProperTime{0.0};
        std::vector<BodyDivergenceRecord> perBody;

        // FieldGrid2D metrics
        double fieldL2Residual{0.0};
        double fieldPeakResidual{0.0};
        double fieldPeakA{0.0};
        double fieldPeakB{0.0};

        // RadialAndScaleProfile metrics
        double curveMaxAbsDiff{0.0};
        double curveRmsDiff{0.0};
        double curveMaxRelDiff{0.0};
    };

    struct ApplicationOptions {
        bool showHelp{false};
        bool showVersion{false};
        bool forceHeadless{false};
        bool profileOnly{false};
        bool dumpHardwareJson{false};

        std::optional<int> preferredGpuDevice{std::nullopt};
        std::optional<std::filesystem::path> scenePath{std::nullopt};
        std::optional<scene::ModelId> overridePrimaryModel{std::nullopt};
        std::optional<scene::ModelId> overrideCompareModel{std::nullopt};
        std::optional<std::uint64_t> overrideSteps{std::nullopt};
        std::optional<double> overrideTimestep{std::nullopt};
        std::optional<std::filesystem::path> outputPath{std::nullopt};

        std::uint32_t windowWidth{1280};
        std::uint32_t windowHeight{720};
        bool explicitWindowSize{false};
        bool forceX11{false};
        std::string logLevel{"info"};
    };

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
            std::optional<scene::ModelId> explicitCompareModel = std::nullopt,
            bool applyCliOverrides = false
        );
        [[nodiscard]] bool StepActiveSolvers(std::uint64_t stepCount, double dt);
        void RefreshDomainSnapshots();
        void UpdateABComparisonMetrics();
        [[nodiscard]] int RunHeadlessSimulation();

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        void RenderDesktopUi();
        void UpdateWindowTitle();
        void CreateNewEmptyScene();
        [[nodiscard]] bool OpenSceneFromPath(const std::filesystem::path& filePath);
        [[nodiscard]] bool SaveActiveSceneToPath(const std::filesystem::path& filePath);
        void AddBodyToActiveScene(const scene::BodyEntity& entity);
        void DeleteBodyFromActiveScene(std::size_t bodyIndex);
        void FocusViewportOnBody(std::size_t bodyIndex);
        void RecordLiveTrails();
#endif

        ApplicationOptions m_options{};
        HardwareCapabilities m_hardwareCaps{};
        HardwareAdaptationPlan m_adaptationPlan{};
        std::unique_ptr<compute::CUDAContext> m_cudaContext;

        scene::Multiverse m_scenarios{};
        scene::Multiverse m_universes{};
        scene::FlowerOfLifeMultiverse m_flowerMultiverse{};

        scene::SimulationScope m_activeScope{scene::SimulationScope::Scenario};
        bool m_hasActiveScene{false};
        scene::UniverseScene m_activeScene{};

        scene::ModelId m_primaryModelId{scene::ModelId::MainstreamNewtonian};
        bool m_compareOverlayEnabled{false};
        scene::ModelId m_compareModelId{scene::ModelId::ThinkerEssonne};

        std::unique_ptr<models::IPhysicsSolver> m_activeSolver;
        std::unique_ptr<models::IPhysicsSolver> m_compareSolver;

        // NBodyDynamics live mirrors
        std::vector<scene::BodyEntity> m_liveBodies;
        std::vector<scene::BodyEntity> m_compareBodies;
        std::vector<double> m_primaryProperTimes;
        std::vector<double> m_compareProperTimes;

        // FieldGrid2D live snapshots & probe waveform history
        models::FieldGrid2DSnapshot m_primaryFieldSnap{};
        models::FieldGrid2DSnapshot m_compareFieldSnap{};
        std::deque<float> m_probeWaveformA;
        std::deque<float> m_probeWaveformB;
        static constexpr std::size_t kMaxWaveformSamples = 200U;

        // RadialAndScaleProfile live snapshots
        models::RadialProfileSnapshot m_primaryRadialSnap{};
        models::RadialProfileSnapshot m_compareRadialSnap{};

        models::SolverDiagnostics m_latestDiagnostics{};
        models::SolverDiagnostics m_compareDiagnostics{};
        ABComparisonMetrics m_abMetrics{};

        double m_activeTimestep{1.0e-3};
        bool m_initialized{false};
        bool m_guiActive{false};

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        std::unique_ptr<graphics::Window> m_window;
        std::unique_ptr<graphics::RenderDevice> m_renderDevice;
        std::unique_ptr<ui::UIManager> m_uiManager;

        /** @brief Explicit legacy-aligned playback state (`Stopped`, `Playing`, `Paused`). */
        std::uint8_t m_playbackStateRaw{1U}; // 0=Stopped, 1=Playing, 2=Paused

        /** @brief Filesystem path of the active scene if opened/saved on disk. */
        std::filesystem::path m_currentScenePath;

        /**
         * @brief When `true`, GUI playback uses a wall-clock fixed-timestep
         *        accumulator so simulation speed is completely independent of
         *        monitor refresh rate / FPS while keeping physics `dt` fixed.
         */
        bool m_wallClockPaced{true};

        /**
         * @brief Simulation-time advancement rate per real wall-clock second
         *        (e.g., `1.0` = exact 1:1 real time on SI-second scenes like C204).
         */
        float m_simTimeScale{1.0f};

        /** @brief Fractional wall-clock accumulator waiting to consume whole `m_activeTimestep` steps. */
        double m_simTimeAccumulator{0.0};

        /** @brief Upper bound on catch-up physics steps executed within a single rendered frame. */
        int m_maxSubStepsPerFrame{256};

        /** @brief Number of fixed-dt physics sub-steps executed on the most recent GUI frame. */
        int m_lastFrameExecutedSteps{0};

        /** @brief Manual per-frame step count used only when `m_wallClockPaced == false`. */
        int m_stepsPerFrame{10};

        std::optional<std::size_t> m_selectedBodyIndex{std::nullopt};
        bool m_entitySelectionJustChanged{false};
        bool m_showFlowerOfLifeLatticeView{false};

        // FieldGrid2D heatmap display selector: 0 = Model A, 1 = Model B, 2 = |A - B| Difference
        int m_fieldHeatmapMode{0};

        bool m_viewportAutoFit{false};
        float m_viewportHalfExtent{6.5f};
        float m_viewportPanX{0.0f};
        float m_viewportPanY{0.0f};

        std::vector<std::deque<scene::Vec3d>> m_bodyTrails;
        std::vector<std::deque<scene::Vec3d>> m_compareTrails;
        static constexpr std::size_t kMaxTrailPoints = 400U;
#endif
    };

} // namespace umgebung::core
