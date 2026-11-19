/**
 * @file Application.hpp
 * @brief Top-level lifecycle orchestrator for the Umgebung Reality Simulation Engine.
 *
 * @details
 * `umgebung::core::Application` coordinates all engine pillars across both
 * Desktop Interactive GUI builds (`UMGEBUNG_HAS_GUI == 1`) and Headless HPC
 * Batch builds (`UMGEBUNG_HAS_GUI == 0`):
 *   1. **CLI Parsing (`cxxopts`):** Supports hardware profiling (`--profile-only`,
 *      `--json-profile`, `--device`), scene/solver control (`--scene`, `--steps`,
 *      `--dt`, `--output`), and desktop GUI options (`--headless`, `--width`,
 *      `--height`, `--x11`, `--log-level`).
 *   2. **Startup Hardware Discovery & Adaptation (`SystemProfiler`):** Probes
 *      CPU, host RAM, and every NVIDIA GPU, deriving a `HardwareAdaptationPlan`.
 *   3. **GPU Compute Initialization (`compute::CUDAContext`):** Binds the
 *      adapted primary GPU, creates non-blocking CUDA streams, and enforces
 *      the adapted VRAM budget.
 *   4. **Multiverse & Scene Loading (`scene::Multiverse`):** Loads `.umgebung`
 *      universe files from `--scene` or auto-discovers bundled `scenes/`.
 *   5. **Mainstream Physics Solver (`models::mainstream::NewtonianNBodySolver`):**
 *      Initializes FP64 SoA GPU buffers for the active universe and drives
 *      either headless batch integration or live interactive stepping.
 *   6. **Conditional Desktop GUI (`Window` + `RenderDevice`, GUI builds only):**
 *      Renders the Vulkan 1.2 + NVRHI + Dear ImGui workspace (Hardware Telemetry,
 *      Multiverse Hierarchy, Simulation Controls/Invariants, and 2D Orbital View).
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/mainstream/NewtonianNBodySolver.hpp"
#include "umgebung/scene/Multiverse.hpp"

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
     * @struct ApplicationOptions
     * @brief Parsed command-line configuration governing a single `Application` run.
     */
    struct ApplicationOptions {
        /** @brief True if the user requested CLI usage output (`-h` / `--help`). */
        bool showHelp{false};

        /** @brief True if the user requested engine version metadata (`-v` / `--version`). */
        bool showVersion{false};

        /** @brief True to suppress GUI window creation even in a GUI-enabled build (`--headless`). */
        bool forceHeadless{false};

        /** @brief True to perform startup hardware profiling/adaptation and immediately exit (`--profile-only`). */
        bool profileOnly{false};

        /** @brief True to print the machine-readable JSON hardware/adaptation report to `stdout` (`--json-profile`). */
        bool dumpHardwareJson{false};

        /** @brief Optional CUDA GPU device index requested via `--device <id>`. */
        std::optional<int> preferredGpuDevice{std::nullopt};

        /** @brief Optional path to a `.umgebung` scene file or directory of scenes (`--scene`). */
        std::optional<std::filesystem::path> scenePath{std::nullopt};

        /** @brief Optional headless batch step count override (`--steps`). */
        std::optional<std::uint64_t> overrideSteps{std::nullopt};

        /** @brief Optional timestep $\Delta t$ override (`--dt`). */
        std::optional<double> overrideTimestep{std::nullopt};

        /** @brief Optional JSON telemetry output path for headless batch runs (`--output`). */
        std::optional<std::filesystem::path> outputPath{std::nullopt};

        /** @brief Requested initial desktop window width in screen coordinates (`--width`). */
        std::uint32_t windowWidth{1280};

        /** @brief Requested initial desktop window height in screen coordinates (`--height`). */
        std::uint32_t windowHeight{720};

        /**
         * @brief On Linux GUI builds, forces the X11/XWayland GLFW backend
         *        instead of defaulting to native Wayland (`--x11`).
         */
        bool forceX11{false};

        /** @brief Requested `spdlog` verbosity level (`trace`, `debug`, `info`, `warn`, `error`, `off`). */
        std::string logLevel{"info"};
    };

    /**
     * @class Application
     * @brief Owns and coordinates the engine startup sequence, hardware
     *        adaptation state, CUDA compute context, multiverse scenes, active
     *        physics solver, and optional GUI `Window` + `RenderDevice`.
     */
    class Application final {
    public:
        /** @brief Semantic version string for the Umgebung engine binary. */
        static constexpr const char* kEngineVersion = "0.0.1";

        /** @brief Constructs an uninitialized `Application`; call `Initialize()` before `Run()`. */
        Application();

        /** @brief Destructor; guarantees deterministic reverse-order shutdown of all subsystems. */
        ~Application();

        Application(const Application&) = delete;
        Application& operator=(const Application&) = delete;
        Application(Application&&) = delete;
        Application& operator=(Application&&) = delete;

        /**
         * @brief Parses CLI flags, configures logging, detects/adapts to host
         *        hardware, initializes `CUDAContext`, loads `.umgebung` scenes,
         *        prepares the active physics solver, and (in GUI mode) creates
         *        `Window` + `RenderDevice`.
         *
         * @param argc Argument count forwarded from `main()`.
         * @param argv Argument vector forwarded from `main()`.
         * @return `ApplicationInitResult::Ready`, `EarlyExitSuccess`, or `Failure`.
         */
        [[nodiscard]] ApplicationInitResult Initialize(int argc, char** argv);

        /**
         * @brief Executes either the interactive GUI render/simulation loop or
         *        the headless HPC batch simulation run.
         *
         * @return Process exit code (`0` on normal completion, non-zero on error).
         */
        [[nodiscard]] int Run();

        /**
         * @brief Releases all owned subsystems in strict dependency order:
         *        `NewtonianNBodySolver` -> `CUDAContext` -> `RenderDevice` -> `Window`.
         */
        void Shutdown() noexcept;

        /** @brief Returns parsed CLI options. */
        [[nodiscard]] const ApplicationOptions& GetOptions() const noexcept { return m_options; }

        /** @brief Returns startup hardware capabilities snapshot. */
        [[nodiscard]] const HardwareCapabilities& GetHardwareCapabilities() const noexcept {
            return m_hardwareCaps;
        }

        /** @brief Returns startup hardware adaptation plan. */
        [[nodiscard]] const HardwareAdaptationPlan& GetAdaptationPlan() const noexcept {
            return m_adaptationPlan;
        }

        /** @brief Returns the active `compute::CUDAContext`, or `nullptr` if uninitialized. */
        [[nodiscard]] compute::CUDAContext* GetCudaContext() const noexcept { return m_cudaContext.get(); }

        /** @brief Returns the loaded `scene::Multiverse` container. */
        [[nodiscard]] const scene::Multiverse& GetMultiverse() const noexcept { return m_multiverse; }

        /** @brief Returns `true` if interactive Desktop GUI mode is active. */
        [[nodiscard]] bool IsGuiActive() const noexcept { return m_guiActive; }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        /** @brief Returns a non-owning pointer to the active desktop `Window`, or `nullptr`. */
        [[nodiscard]] graphics::Window* GetWindow() const noexcept { return m_window.get(); }

        /** @brief Returns a non-owning pointer to the active `RenderDevice`, or `nullptr`. */
        [[nodiscard]] graphics::RenderDevice* GetRenderDevice() const noexcept { return m_renderDevice.get(); }
#endif

    private:
        [[nodiscard]] ApplicationInitResult ParseCommandLine(int argc, char** argv);
        void ConfigureLogging() const;
        [[nodiscard]] bool LoadInitialMultiverse(const char* argv0);
        [[nodiscard]] bool ActivateUniverseForSimulation(std::size_t universeIndex);
        [[nodiscard]] int RunHeadlessSimulation();

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        void RenderDesktopUi();
        void RenderBodyHierarchyNode(
            const std::vector<scene::BodyEntity>& bodies,
            const std::string& parentId
        );
        void RecordLiveTrails();
#endif

        /** @brief Parsed CLI configuration. */
        ApplicationOptions m_options{};

        /** @brief Startup hardware snapshot. */
        HardwareCapabilities m_hardwareCaps{};

        /** @brief Startup hardware adaptation plan. */
        HardwareAdaptationPlan m_adaptationPlan{};

        /** @brief Owned GPU compute context (streams + VRAM budget tracker). */
        std::unique_ptr<compute::CUDAContext> m_cudaContext;

        /** @brief Loaded collection of `.umgebung` universe scenes. */
        scene::Multiverse m_multiverse{};

        /** @brief Owned Mainstream FP64 CUDA Newtonian N-body solver (destroyed before `m_cudaContext`). */
        std::unique_ptr<models::mainstream::NewtonianNBodySolver> m_nbodySolver;

        /** @brief Host mirror of the active universe's live body positions/velocities for UI/output. */
        std::vector<scene::BodyEntity> m_liveBodies;

        /** @brief Latest conservation-law & energy diagnostics snapshot from `m_nbodySolver`. */
        models::mainstream::NBodyDiagnostics m_latestDiagnostics{};

        /** @brief Active timestep $\Delta t$ for the currently loaded universe. */
        double m_activeTimestep{1.0e-3};

        /** @brief True after `Initialize()` succeeds. */
        bool m_initialized{false};

        /** @brief True when GUI windowing/rendering is both compiled in and active. */
        bool m_guiActive{false};

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        /** @brief Owned desktop OS window (exclusive to GUI builds). */
        std::unique_ptr<graphics::Window> m_window;

        /** @brief Owned Vulkan 1.2 + NVRHI + Dear ImGui renderer (destroyed before `m_window`). */
        std::unique_ptr<graphics::RenderDevice> m_renderDevice;

        /** @brief Interactive simulation play/pause toggle in GUI mode. */
        bool m_simulationRunning{true};

        /** @brief Number of CUDA integration sub-steps executed per rendered GUI frame. */
        int m_stepsPerFrame{10};

        /** @brief Visibility toggles for dockable ImGui workspace panels. */
        bool m_showHardwarePanel{true};
        bool m_showHierarchyPanel{true};
        bool m_showSimulationPanel{true};
        bool m_showViewport2DPanel{true};

        /** @brief Per-body recent XY orbital trail history rendered in the 2D Orbital Viewport. */
        std::vector<std::deque<scene::Vec3d>> m_bodyTrails;

        /** @brief Maximum trail vertices retained per body in the 2D Orbital Viewport. */
        static constexpr std::size_t kMaxTrailPoints = 400U;
#endif
    };

} // namespace umgebung::core
