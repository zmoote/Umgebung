/**
 * @file Application.hpp
 * @brief Top-level lifecycle orchestrator for the Umgebung Reality Simulation Engine.
 *
 * @details
 * `umgebung::core::Application` is the single high-level entry controller
 * shared by both Desktop Interactive GUI builds (`UMGEBUNG_HAS_GUI == 1`) and
 * Headless HPC Batch builds (`UMGEBUNG_HAS_GUI == 0`).
 *
 * Startup and execution sequence:
 *   1. **CLI Parsing (`cxxopts`):** Parses runtime flags (`--help`, `--version`,
 *      `--headless`, `--profile-only`, `--json-profile`, `--device`, `--width`,
 *      `--height`, `--log-level`).
 *   2. **Logging Initialization (`spdlog`):** Configures verbosity and clean
 *      timestamped output formatting.
 *   3. **Hardware Discovery & Adaptation (`SystemProfiler`):** Probes host CPU,
 *      physical/available RAM, and every visible NVIDIA CUDA GPU, computes a
 *      safe `HardwareAdaptationPlan` tuned to the active execution mode, binds
 *      the selected primary CUDA device via `cudaSetDevice`, and logs (or
 *      exports as JSON) the full startup profile.
 *   4. **Conditional Desktop Window Creation (`Window`, GUI builds only):**
 *      When compiled with `UMGEBUNG_HAS_GUI == 1` and neither `--headless` nor
 *      `--profile-only` is specified, instantiates and initializes the dedicated
 *      `umgebung::graphics::Window` controller.
 *   5. **Main Loop & Deterministic Shutdown:** Drives either the interactive
 *      window event loop or headless batch execution, then guarantees RAII
 *      teardown of all owned subsystems.
 */

#pragma once

#include "umgebung/core/SystemProfiler.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

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
     *
     * @details
     * Distinguishes normal interactive/batch readiness from intentional early
     * exits (such as `--help` or `--version`, which should exit with status `0`
     * without entering `Run()`) and genuine initialization errors (exit non-zero).
     */
    enum class ApplicationInitResult : std::uint8_t {
        /** @brief All startup stages succeeded; caller should invoke `Run()`. */
        Ready = 0,
        /** @brief Handled an informational flag (`--help` / `--version`); exit with `0`. */
        EarlyExitSuccess = 1,
        /** @brief Startup failed (invalid CLI, missing CUDA GPU, or window init failure). */
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
     *        adaptation state, optional GUI `Window`, and top-level run loop.
     */
    class Application final {
    public:
        /** @brief Semantic version string for the Umgebung engine binary. */
        static constexpr const char* kEngineVersion = "0.0.1";

        /** @brief Constructs an uninitialized `Application`; call `Initialize()` before `Run()`. */
        Application();

        /** @brief Destructor; guarantees deterministic shutdown of window and engine resources. */
        ~Application();

        // Application represents a unique process-level runtime instance.
        Application(const Application&) = delete;
        Application& operator=(const Application&) = delete;
        Application(Application&&) = delete;
        Application& operator=(Application&&) = delete;

        /**
         * @brief Parses CLI flags, configures logging, detects and adapts to
         *        host/GPU hardware, and (in GUI mode) opens the desktop window.
         *
         * @param argc Argument count forwarded from `main()`.
         * @param argv Argument vector forwarded from `main()`.
         * @return `ApplicationInitResult::Ready`, `EarlyExitSuccess`, or `Failure`.
         */
        [[nodiscard]] ApplicationInitResult Initialize(int argc, char** argv);

        /**
         * @brief Executes the active runtime loop (interactive GUI window loop
         *        or headless compute session) until completion or close request.
         *
         * @return Process exit code (`0` on normal completion, non-zero on runtime failure).
         * @pre    `Initialize()` returned `ApplicationInitResult::Ready`.
         */
        [[nodiscard]] int Run();

        /**
         * @brief Releases all owned subsystems (GUI `Window`, runtime state) in
         *        reverse initialization order. Safe to call multiple times.
         */
        void Shutdown() noexcept;

        /** @brief Returns the parsed CLI options for this application instance. */
        [[nodiscard]] const ApplicationOptions& GetOptions() const noexcept { return m_options; }

        /** @brief Returns the startup hardware snapshot discovered by `SystemProfiler`. */
        [[nodiscard]] const HardwareCapabilities& GetHardwareCapabilities() const noexcept {
            return m_hardwareCaps;
        }

        /** @brief Returns the startup hardware adaptation plan derived from `m_hardwareCaps`. */
        [[nodiscard]] const HardwareAdaptationPlan& GetAdaptationPlan() const noexcept {
            return m_adaptationPlan;
        }

        /** @brief Returns `true` if the application is actively running in Desktop GUI mode. */
        [[nodiscard]] bool IsGuiActive() const noexcept { return m_guiActive; }

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        /**
         * @brief Returns a non-owning pointer to the active desktop `Window`,
         *        or `nullptr` if running in `--headless` / `--profile-only` mode.
         */
        [[nodiscard]] graphics::Window* GetWindow() const noexcept { return m_window.get(); }

        /**
         * @brief Returns a non-owning pointer to the active Vulkan/NVRHI/ImGui
         *        `RenderDevice`, or `nullptr` in `--headless` / `--profile-only` mode.
         */
        [[nodiscard]] graphics::RenderDevice* GetRenderDevice() const noexcept { return m_renderDevice.get(); }
#endif

    private:
        /**
         * @brief Parses command-line options into `m_options` using `cxxopts`.
         *
         * @param argc Argument count from `main()`.
         * @param argv Argument vector from `main()`.
         * @return `Ready` if parsing succeeded and normal startup should proceed,
         *         `EarlyExitSuccess` for `--help`/`--version`, or `Failure` on syntax errors.
         */
        [[nodiscard]] ApplicationInitResult ParseCommandLine(int argc, char** argv);

        /**
         * @brief Configures `spdlog`'s global log pattern and verbosity level
         *        from `m_options.logLevel`.
         */
        void ConfigureLogging() const;

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        /**
         * @brief Emits the interactive Dear ImGui dockspace, hardware/adaptation
         *        telemetry panel, and multiverse hierarchy inspector for one frame.
         */
        void RenderDesktopUi();
#endif

        /** @brief Parsed CLI configuration for the current run. */
        ApplicationOptions m_options{};

        /** @brief Immutable snapshot of host CPU, host RAM, and all NVIDIA CUDA GPUs. */
        HardwareCapabilities m_hardwareCaps{};

        /** @brief Adaptive runtime policy (GPU selection, VRAM/RAM/thread budgets, launch geometry). */
        HardwareAdaptationPlan m_adaptationPlan{};

        /** @brief True if `Initialize()` completed normally and `Shutdown()` has not yet finalized. */
        bool m_initialized{false};

        /** @brief True when GUI windowing is both compiled in (`UMGEBUNG_HAS_GUI == 1`) and active for this run. */
        bool m_guiActive{false};

#if defined(UMGEBUNG_HAS_GUI) && (UMGEBUNG_HAS_GUI == 1)
        /** @brief Owned desktop OS window instance (exclusive to GUI builds). */
        std::unique_ptr<graphics::Window> m_window;

        /** @brief Owned Vulkan 1.2 + NVRHI + Dear ImGui renderer (destroyed before `m_window`). */
        std::unique_ptr<graphics::RenderDevice> m_renderDevice;

        /** @brief Visibility toggle for the Hardware & Adaptation Telemetry ImGui panel. */
        bool m_showHardwarePanel{true};

        /** @brief Visibility toggle for the Multiverse Hierarchy ImGui panel. */
        bool m_showHierarchyPanel{true};
#endif
    };

} // namespace umgebung::core
