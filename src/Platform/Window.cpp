// --- Platform/Window.cpp ---
#include "Window.hpp"
#include <GLFW/glfw3.h>
#include "../Core/Logger.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace Umgebung {

    void Window::Create(const std::string& title, uint32_t width, uint32_t height) {
        if (!glfwInit()) {
            Logger::GetCoreLogger()->error("Failed to initialize GLFW.");
            return;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // <<<< REQUIRED for Vulkan

        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!window) {
            Logger::GetCoreLogger()->error("Failed to create GLFW window.");
            glfwTerminate();
        }

        SetIcon("resources/Umgebung.png");

        glfwSetWindowUserPointer(window, this);

        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, [[maybe_unused]]int width, [[maybe_unused]]int height) {
            auto* wnd = static_cast<Window*>(glfwGetWindowUserPointer(win));
            if (wnd) {
                wnd->framebufferResized = true;
            }
            });
    }

    void Window::PollEvents() {
        glfwPollEvents();
    }

    GLFWwindow* Window::GetNativeWindow() {
        return window;
    }

    bool Window::ShouldClose() const {
        return glfwWindowShouldClose(window);
    }

    void Window::SetIcon(const std::string& path) {
        GLFWimage icon;
        icon.pixels = stbi_load(path.c_str(), &icon.width, &icon.height, 0, 4); // Force RGBA
        if (icon.pixels) {
            glfwSetWindowIcon(window, 1, &icon);
            stbi_image_free(icon.pixels);
        }
        else {
            Umgebung::Logger::GetCoreLogger()->warn("Failed to load icon image: {}", path);
        }
    }

    bool Window::WasResized() const {
        return framebufferResized;
    }

    void Window::ResetResizedFlag() {
        framebufferResized = false;
    }

} // namespace Umgebung
