/**
 * @file Window.cpp
 * @brief Implements the Window class.
 */
#include "umgebung/ui/Window.hpp"
#include "umgebung/util/LogMacros.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace Umgebung::ui {

    /**
     * @brief The GLFW error callback function.
     * 
     * @param error The error code.
     * @param description The error description.
     */
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

        int iconWidth, iconHeight, iconChannels;
        stbi_uc* pixels = stbi_load("assets/icon/Umgebung.png", &iconWidth, &iconHeight, &iconChannels, 4);
        if (pixels) {
            GLFWimage images[1];
            images[0].width = iconWidth;
            images[0].height = iconHeight;
            images[0].pixels = pixels;

            glfwSetWindowIcon(m_window, 1, images);

            stbi_image_free(pixels);
        }

        glfwSetWindowUserPointer(m_window, this);

        glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);

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

        if (resizeCallback_) {
            resizeCallback_(width, height);
        }
    }

    void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (self) {
            self->onResize(width, height);
        }
    }
}