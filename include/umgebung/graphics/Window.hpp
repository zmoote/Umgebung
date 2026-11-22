/**
 * @file Window.hpp
 * @brief Dedicated OS window abstraction exclusive to Desktop Interactive GUI builds.
 *
 * @details
 * Provides an RAII-managed wrapper around GLFW window creation, display-server
 * backend selection (Windows Win32, Linux X11/XWayland, Linux Wayland), event
 * polling, framebuffer resize tracking, and Vulkan instance-extension querying.
 *
 * Separation of concerns:
 *   - `umgebung::graphics::Window` owns **only** the OS window and input/event
 *     pump. It does **not** create Vulkan instances, NVRHI devices, swapchains,
 *     or Dear ImGui contexts; those belong to dedicated rendering classes that
 *     attach to `Window` via `GetNativeHandle()` and `GetRequiredVulkanExtensions()`.
 *   - This header and its corresponding implementation (`src/graphics/Window.cpp`)
 *     are compiled **strictly** when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 *     Headless HPC builds never include or link this module.
 *
 * @warning Attempting to include this header when `UMGEBUNG_HAS_GUI == 0`
 *          triggers a compile-time `#error` to prevent accidental GUI leakage
 *          into headless HPC translation units.
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/graphics/Window.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/**
 * @brief Forward declaration of the opaque GLFW window handle.
 *
 * @details
 * Kept forward-declared in the public header so `<GLFW/glfw3.h>` and platform
 * windowing macros never pollute unrelated engine headers.
 */
struct GLFWwindow;

/**
 * @namespace umgebung::graphics
 * @brief Desktop-only windowing, Vulkan/NVRHI rendering, and Dear ImGui UI
 *        subsystems compiled exclusively when `UMGEBUNG_BUILD_GUI=ON`.
 */
namespace umgebung::graphics {

    /**
     * @struct WindowConfig
     * @brief Immutable startup configuration parameters for constructing an
     *        interactive desktop `Window`.
     */
    struct WindowConfig {
        /** @brief UTF-8 window title displayed in the OS title bar and taskbar. */
        std::string title{"Umgebung - Reality Simulation Engine"};

        /** @brief Initial client-area width in screen coordinates (pixels at 100% DPI). */
        std::uint32_t width{1280};

        /** @brief Initial client-area height in screen coordinates (pixels at 100% DPI). */
        std::uint32_t height{720};

        /** @brief True to allow interactive resizing and maximizing by the user. */
        bool resizable{true};

        /**
         * @brief True to launch the desktop window maximized on startup (matching
         *        legacy Umgebung's full-workstation startup experience).
         */
        bool maximized{false};

        /**
         * @brief On Linux desktop environments, prefers X11/XWayland when `DISPLAY`
         *        is set so Dear ImGui multi-viewport panel detachment outside the
         *        main window works out of the box.
         *
         * @details
         * Defaults to `true` because native Wayland forbids global window
         * positioning (`glfwSetWindowPos`), causing Dear ImGui to disable
         * multi-viewport detachment. Set `preferWaylandOnLinux = true` (via
         * `--wayland`) to opt into native Wayland when multi-viewport dragging
         * is not needed. Pure Wayland sessions with `DISPLAY` unset still use
         * native Wayland automatically.
         */
        bool preferX11OnLinux{true};

        /**
         * @brief On Linux desktop environments, forces GLFW's native Wayland
         *        backend whenever `WAYLAND_DISPLAY` is set (`--wayland` CLI flag).
         */
        bool preferWaylandOnLinux{false};
    };

    /**
     * @class Window
     * @brief RAII OS window controller backed by GLFW (`GLFW_NO_API` for Vulkan/NVRHI).
     *
     * @details
     * Instances are non-copyable and non-movable because GLFW stores a raw
     * `this` user-pointer (`glfwSetWindowUserPointer`) to route asynchronous OS
     * callbacks (such as framebuffer resize notifications) back to the owning
     * `Window` instance.
     */
    class Window final {
    public:
        /** @brief Signature invoked whenever the OS reports a framebuffer size change. */
        using ResizeCallback = std::function<void(std::uint32_t width, std::uint32_t height)>;

        /** @brief Constructs an uninitialized `Window`; call `Initialize()` before use. */
        Window() = default;

        /** @brief Destructor; guarantees deterministic GLFW window & subsystem teardown. */
        ~Window();

        // Non-copyable and non-movable to keep GLFW user-pointer addresses stable.
        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;
        Window(Window&&) = delete;
        Window& operator=(Window&&) = delete;

        /**
         * @brief Initializes GLFW, validates Vulkan WSI availability, and creates
         *        the OS window with `GLFW_CLIENT_API = GLFW_NO_API`.
         *
         * @param config Desired window dimensions, title, and Linux backend hints.
         * @return `true` if the window was created successfully; `false` on any
         *         display-server, GLFW, or Vulkan-support failure (details logged via `spdlog`).
         *
         * @pre  `IsInitialized()` is `false` (calling `Initialize()` twice returns `false`).
         * @post On `true`, `GetNativeHandle() != nullptr` and framebuffer extents are populated.
         */
        [[nodiscard]] bool Initialize(const WindowConfig& config);

        /**
         * @brief Destroys the native GLFW window and terminates GLFW if owned.
         *
         * @details
         * Idempotent: safe to invoke explicitly during `Application::Shutdown()`
         * and again automatically inside `~Window()`.
         */
        void Shutdown() noexcept;

        /**
         * @brief Processes all pending OS window/input events without blocking.
         *
         * @details
         * Intended to be called once per frame inside the interactive desktop
         * main loop.
         */
        void PollEvents() const;

        /**
         * @brief Blocks the calling thread until at least one OS event arrives.
         *
         * @details
         * Useful when the window is minimized (`IsMinimized() == true`) so the
         * render loop avoids spinning at 100% CPU/GPU while hidden.
         */
        void WaitEvents() const;

        /**
         * @brief Queries whether the user or OS has requested window closure
         *        (e.g., clicking the close button or pressing Alt+F4).
         *
         * @return `true` if the window should close or is not initialized; `false` otherwise.
         */
        [[nodiscard]] bool ShouldClose() const noexcept;

        /**
         * @brief Programmatically marks the window for closure on the next loop check.
         */
        void RequestClose() noexcept;

        /**
         * @brief Updates the OS title bar text at runtime (e.g., showing active
         *        GPU name, physics profile, or live FPS/simulation step).
         *
         * @param title New UTF-8 window title string.
         */
        void SetTitle(const std::string& title);

        /**
         * @brief Registers a callback invoked immediately whenever the window's
         *        drawable framebuffer dimensions change.
         *
         * @param callback Callable receiving `(newWidthPixels, newHeightPixels)`.
         */
        void SetResizeCallback(ResizeCallback callback);

        /**
         * @brief Checks and clears the sticky framebuffer-resized flag.
         *
         * @return `true` if a resize occurred since the previous call; `false` otherwise.
         */
        [[nodiscard]] bool ConsumeResizeFlag() noexcept;

        /** @brief Returns `true` if the OS window has been initialized and is alive. */
        [[nodiscard]] bool IsInitialized() const noexcept { return m_window != nullptr; }

        /** @brief Returns `true` when either framebuffer dimension is zero (window iconified/minimized). */
        [[nodiscard]] bool IsMinimized() const noexcept {
            return (m_framebufferWidth == 0U) || (m_framebufferHeight == 0U);
        }

        /** @brief Returns the current drawable framebuffer width in physical pixels. */
        [[nodiscard]] std::uint32_t GetFramebufferWidth() const noexcept { return m_framebufferWidth; }

        /** @brief Returns the current drawable framebuffer height in physical pixels. */
        [[nodiscard]] std::uint32_t GetFramebufferHeight() const noexcept { return m_framebufferHeight; }

        /** @brief Returns the underlying `GLFWwindow*` for Vulkan surface creation and ImGui bindings. */
        [[nodiscard]] GLFWwindow* GetNativeHandle() const noexcept { return m_window; }

        /**
         * @brief Queries GLFW for the Vulkan instance extensions required to
         *        create a presentation surface on the active display server.
         *
         * @return Vector of static extension name strings (empty if unavailable).
         */
        [[nodiscard]] std::vector<const char*> GetRequiredVulkanExtensions() const;

    private:
        /**
         * @brief Static C-ABI trampoline registered with `glfwSetFramebufferSizeCallback`.
         *
         * @param handle Native GLFW window handle that experienced the resize.
         * @param width  New drawable framebuffer width in physical pixels.
         * @param height New drawable framebuffer height in physical pixels.
         */
        static void OnFramebufferResizeCallback(GLFWwindow* handle, int width, int height);

        /** @brief Owned native GLFW window handle (`nullptr` when uninitialized or shut down). */
        GLFWwindow* m_window{nullptr};

        /** @brief Tracks whether this instance successfully invoked `glfwInit()` and must call `glfwTerminate()`. */
        bool m_glfwInitialized{false};

        /** @brief Current cached drawable framebuffer width in physical pixels. */
        std::uint32_t m_framebufferWidth{0};

        /** @brief Current cached drawable framebuffer height in physical pixels. */
        std::uint32_t m_framebufferHeight{0};

        /** @brief Sticky flag set by `OnFramebufferResizeCallback` and cleared by `ConsumeResizeFlag()`. */
        bool m_framebufferResized{false};

        /** @brief Optional user/renderer callback triggered on framebuffer size changes. */
        ResizeCallback m_resizeCallback;
    };

} // namespace umgebung::graphics
