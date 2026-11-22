/**
 * @file Window.cpp
 * @brief Implementation of the GUI-exclusive GLFW OS window controller.
 *
 * @details
 * Implements `umgebung::graphics::Window` with explicit diagnostics for Linux
 * X11/XWayland vs. native Wayland environments, strict GLFW error forwarding
 * into `spdlog`, and clean RAII teardown so no display handles leak on early
 * initialization failure.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/graphics/Window.hpp"

#include <cstdlib>
#include <utility>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace umgebung::graphics {

    namespace {

        /**
         * @brief Global GLFW error callback routing internal GLFW failures directly
         *        into Umgebung's `spdlog` logging sink.
         *
         * @param errorCode   Numeric GLFW error code (e.g., `GLFW_PLATFORM_ERROR`).
         * @param description Human-readable UTF-8 diagnostic provided by GLFW.
         */
        void GlfwErrorCallback(int errorCode, const char* description) {
            spdlog::error(
                "GLFW Error [0x{:08X}]: {}",
                static_cast<unsigned int>(errorCode),
                description ? description : "(no description)"
            );
        }

        /**
         * @brief Translates a GLFW runtime platform identifier into a readable string.
         *
         * @param platform Value returned by `glfwGetPlatform()`.
         * @return Static string literal naming the active windowing backend.
         */
        [[nodiscard]] const char* GlfwPlatformToString(int platform) noexcept {
            switch (platform) {
                case GLFW_PLATFORM_WIN32:
                    return "Win32";
                case GLFW_PLATFORM_COCOA:
                    return "Cocoa";
                case GLFW_PLATFORM_WAYLAND:
                    return "Wayland";
                case GLFW_PLATFORM_X11:
                    return "X11 / XWayland";
                case GLFW_PLATFORM_NULL:
                    return "Null";
                default:
                    return "Unknown";
            }
        }

    } // anonymous namespace

    Window::~Window() {
        // Guarantee deterministic OS window and GLFW teardown even on stack unwind.
        Shutdown();
    }

    bool Window::Initialize(const WindowConfig& config) {
        if (m_window != nullptr || m_glfwInitialized) {
            spdlog::warn("Window::Initialize called on an already initialized Window instance.");
            return false;
        }

        spdlog::info(
            "Initializing Desktop Window ({}x{}, title=\"{}\")...",
            config.width,
            config.height,
            config.title
        );

        // Route all GLFW diagnostics into spdlog before invoking any other GLFW API.
        glfwSetErrorCallback(GlfwErrorCallback);

#if defined(UMGEBUNG_PLATFORM_LINUX)
        // Inspect Linux display-server environment variables to aid debugging
        // across KDE Plasma Wayland, GNOME Wayland, and X11 desktop sessions.
        const char* displayEnv = std::getenv("DISPLAY");
        const char* waylandEnv = std::getenv("WAYLAND_DISPLAY");
        const char* xdgRuntimeEnv = std::getenv("XDG_RUNTIME_DIR");

        spdlog::info(
            "Linux Display Environment -> DISPLAY: {}, WAYLAND_DISPLAY: {}, XDG_RUNTIME_DIR: {}",
            displayEnv ? displayEnv : "unset",
            waylandEnv ? waylandEnv : "unset",
            xdgRuntimeEnv ? xdgRuntimeEnv : "unset"
        );

        // Backend selection policy on Linux:
        //   1. If the user explicitly requested `--x11` (`preferX11OnLinux == true`)
        //      and `DISPLAY` is set, select X11/XWayland (enables detached Dear ImGui
        //      multi-viewports across external displays on Wayland sessions).
        //   2. Otherwise, use the host's native display server: native Wayland
        //      whenever `WAYLAND_DISPLAY` is set, falling back to native X11 on
        //      X11-only sessions.
        const bool hasDisplay = (displayEnv != nullptr && displayEnv[0] != '\0');
        const bool hasWayland = (waylandEnv != nullptr && waylandEnv[0] != '\0');

        if (config.preferX11OnLinux && hasDisplay) {
            if (glfwPlatformSupported(GLFW_PLATFORM_X11) == GLFW_TRUE) {
                glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
                spdlog::info("Selected GLFW X11/XWayland platform backend (--x11 override).");
            } else {
                spdlog::warn("X11 backend requested via --x11, but not supported by this GLFW build; using auto-detect.");
            }
        } else if (hasWayland) {
            if (glfwPlatformSupported(GLFW_PLATFORM_WAYLAND) == GLFW_TRUE) {
                glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
                spdlog::info("Selected GLFW native Wayland platform backend (host default).");
            } else {
                spdlog::warn("WAYLAND_DISPLAY is set, but Wayland is not supported by this GLFW build; using auto-detect.");
            }
        } else if (hasDisplay) {
            if (glfwPlatformSupported(GLFW_PLATFORM_X11) == GLFW_TRUE) {
                glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
                spdlog::info("Selected GLFW native X11 platform backend (WAYLAND_DISPLAY unset).");
            }
        }
#endif

        if (glfwInit() != GLFW_TRUE) {
            spdlog::error("Failed to initialize GLFW. Verify that a desktop display server is active.");
            return false;
        }
        m_glfwInitialized = true;

        spdlog::info(
            "GLFW initialized (version {}, backend: {}).",
            glfwGetVersionString(),
            GlfwPlatformToString(glfwGetPlatform())
        );

        // Verify that the host system exposes a usable Vulkan loader/ICD for NVRHI.
        if (glfwVulkanSupported() != GLFW_TRUE) {
            spdlog::error("GLFW reports that Vulkan is not supported on this system.");
            Shutdown();
            return false;
        }

        // Disable OpenGL/OpenGL ES context creation because Umgebung renders via
        // Vulkan + NVIDIA NVRHI.
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_MAXIMIZED, config.maximized ? GLFW_TRUE : GLFW_FALSE);

        m_window = glfwCreateWindow(
            static_cast<int>(config.width),
            static_cast<int>(config.height),
            config.title.c_str(),
            nullptr,
            nullptr
        );
        // Reset sticky GLFW_MAXIMIZED hint immediately so secondary OS windows
        // created later by Dear ImGui multi-viewports never auto-maximize!
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE);
        if (m_window == nullptr) {
            spdlog::error("Failed to create GLFW desktop window.");
            Shutdown();
            return false;
        }

        // Bind `this` to the native window so static C callbacks can update state.
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, OnFramebufferResizeCallback);

        // Query actual physical framebuffer pixel extents (accounts for HiDPI scaling).
        int fbWidth = 0;
        int fbHeight = 0;
        glfwGetFramebufferSize(m_window, &fbWidth, &fbHeight);
        m_framebufferWidth = (fbWidth > 0) ? static_cast<std::uint32_t>(fbWidth) : 0U;
        m_framebufferHeight = (fbHeight > 0) ? static_cast<std::uint32_t>(fbHeight) : 0U;
        m_framebufferResized = false;

        spdlog::info(
            "Desktop Window created successfully (framebuffer={}x{} px).",
            m_framebufferWidth,
            m_framebufferHeight
        );
        return true;
    }

    void Window::Shutdown() noexcept {
        if (m_window != nullptr) {
            glfwSetFramebufferSizeCallback(m_window, nullptr);
            glfwSetWindowUserPointer(m_window, nullptr);
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }

        if (m_glfwInitialized) {
            glfwTerminate();
            m_glfwInitialized = false;
            spdlog::info("Desktop Window shut down cleanly.");
        }

        m_framebufferWidth = 0U;
        m_framebufferHeight = 0U;
        m_framebufferResized = false;
    }

    void Window::PollEvents() const {
        if (m_glfwInitialized) {
            glfwPollEvents();
        }
    }

    void Window::WaitEvents() const {
        if (m_glfwInitialized) {
            glfwWaitEvents();
        }
    }

    bool Window::ShouldClose() const noexcept {
        if (m_window == nullptr) {
            return true;
        }
        return glfwWindowShouldClose(m_window) == GLFW_TRUE;
    }

    void Window::RequestClose() noexcept {
        if (m_window != nullptr) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
        }
    }

    void Window::SetTitle(const std::string& title) {
        if (m_window != nullptr) {
            glfwSetWindowTitle(m_window, title.c_str());
        }
    }

    void Window::SetResizeCallback(ResizeCallback callback) {
        m_resizeCallback = std::move(callback);
    }

    bool Window::ConsumeResizeFlag() noexcept {
        const bool wasResized = m_framebufferResized;
        m_framebufferResized = false;
        return wasResized;
    }

    std::vector<const char*> Window::GetRequiredVulkanExtensions() const {
        std::vector<const char*> extensions;
        if (!m_glfwInitialized) {
            return extensions;
        }

        std::uint32_t count = 0U;
        const char** rawExtensions = glfwGetRequiredInstanceExtensions(&count);
        if (rawExtensions != nullptr && count > 0U) {
            extensions.assign(rawExtensions, rawExtensions + count);
        }
        return extensions;
    }

    void Window::OnFramebufferResizeCallback(GLFWwindow* handle, int width, int height) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(handle));
        if (self == nullptr) {
            return;
        }

        self->m_framebufferWidth = (width > 0) ? static_cast<std::uint32_t>(width) : 0U;
        self->m_framebufferHeight = (height > 0) ? static_cast<std::uint32_t>(height) : 0U;
        self->m_framebufferResized = true;

        spdlog::debug(
            "Desktop Window framebuffer resized to {}x{} px.",
            self->m_framebufferWidth,
            self->m_framebufferHeight
        );

        if (self->m_resizeCallback) {
            self->m_resizeCallback(self->m_framebufferWidth, self->m_framebufferHeight);
        }
    }

} // namespace umgebung::graphics
