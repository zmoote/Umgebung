// Window.hpp
#pragma once

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <functional>
#include <string>
#include "util/Logger.hpp"

namespace Umgebung {
    namespace ui {
        class Window {
        public:
            using EventCallbackFn = std::function<void(int)>;   // e.g. key, mouse events

            Window(int width, int height, const std::string& title);
            ~Window();

            // No copying – window handles are unique
            Window(const Window&) = delete;
            Window& operator=(const Window&) = delete;

            // Move‑able for container support
            Window(Window&& other) noexcept;
            Window& operator=(Window&& other) noexcept;

            // Main loop helpers
            void pollEvents() { glfwPollEvents(); }
            bool shouldClose() const { return glfwWindowShouldClose(m_window); }
            void swapBuffers() const { glfwSwapBuffers(m_window); }
            void makeContextCurrent() const { glfwMakeContextCurrent(m_window); }

            // ImGui helpers
            void newFrame();
            void renderImGui();
            ImGuiIO& io() const { return ImGui::GetIO(); }

            // Size accessors
            int width()  const { return m_width; }
            int height() const { return m_height; }

            // Generic user callbacks
            void setEventCallback(const EventCallbackFn& fn) { m_eventCallback = fn; }

        private:
            static void glfwErrorCallback(int code, const char* description);
            static void glfwWindowSizeCallback(GLFWwindow* win, int w, int h);
            static void glfwKeyCallback(GLFWwindow* win, int key, int scancode, int action, int mods);

            GLFWwindow* m_window{ nullptr };
            int           m_width{ 800 };
            int           m_height{ 600 };
            std::string   m_title;
            EventCallbackFn m_eventCallback;   // forwarded to GLFW callbacks
        };

    } // namespace ui
} // namespace Umgebung