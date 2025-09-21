#include "umgebung/ui/Window.hpp"
#include "umgebung/util/LogMacros.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace Umgebung::ui {

    static void glfw_error_callback(int error, const char* description) {
        UMGEBUNG_LOG_ERROR("GLFW Error ({}): {}", error, description);
    }

    Window::Window(int width, int height, const std::string& title)
        : m_width(width), m_height(height), m_title(title) {
    }

    Window::~Window() {
        if (m_window) {
            glfwDestroyWindow(m_window);
        }
        glfwTerminate();
    }

    int Window::init() {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) {
            UMGEBUNG_LOG_CRIT("Failed to initialize GLFW");
            return -1;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_MAXIMIZED, true);

        m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, NULL);
        if (!m_window) {
            UMGEBUNG_LOG_CRIT("Failed to create GLFW window");
            glfwTerminate();
            return -1;
        }

        // --- ADD THIS BLOCK ---
        // This is the missing piece. It connects GLFW's resize event to our class.
        // 1. Store a pointer to this Window instance so the static function can find it.
        glfwSetWindowUserPointer(m_window, this);
        // 2. Tell GLFW to call our static function whenever the window is resized.
        glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
        // ----------------------

        glfwMakeContextCurrent(m_window);
        glfwSwapInterval(1);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            UMGEBUNG_LOG_CRIT("Failed to initialize GLAD");
            return -1;
        }
        UMGEBUNG_LOG_INFO("GLAD Initialized. OpenGL Version: {}", (const char*)glGetString(GL_VERSION));

        glEnable(GL_DEPTH_TEST);

        return 0;
    }

    bool Window::shouldClose() const {
        return glfwWindowShouldClose(m_window);
    }

    void Window::clear() const {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Window::beginFrame() {
        glfwPollEvents();
    }

    void Window::endFrame() {
        glfwSwapBuffers(m_window);
    }

    void Window::beginImGuiFrame() {
        
    }

    void Window::endImGuiFrame() {
       
    }

    void Window::setResizeCallback(const ResizeCallbackFn& callback) {
        resizeCallback_ = callback;
    }

    void Window::onResize(int width, int height) {
        m_width = width;
        m_height = height;

        // Call our stored callback function if it's valid
        if (resizeCallback_) {
            resizeCallback_(width, height);
        }
    }

    // --- ADD THIS MISSING FUNCTION ---
    // This is the C-style function that GLFW calls.
    void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        // 1. Get the pointer to our Window instance that we stored earlier.
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self) {
            // 2. Call the actual C++ member function with the resize logic.
            self->onResize(width, height);
        }
    }
} // namespace Umgebung::ui