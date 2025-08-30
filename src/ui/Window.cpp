#include "umgebung/ui/Window.hpp"
#include "umgebung/util/LogMacros.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace Umgebung {
    namespace ui {
        static void glfw_error_callback(int error, const char* description) {
            UMGEBUNG_LOG_ERROR("GLFW Error ({}): {}", error, description);
        }

        Window::Window(int width, int height, const std::string& title)
            : m_width(width), m_height(height), m_title(title) {
        }

        Window::~Window() {
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            if (m_window) {
                glfwDestroyWindow(m_window);
            }
            glfwTerminate();
            UMGEBUNG_LOG_INFO("Window and GLFW terminated.");
        }

        int Window::init() {
            glfwSetErrorCallback(glfw_error_callback);
            if (!glfwInit()) {
                UMGEBUNG_LOG_CRIT("Failed to initialize GLFW");
                return -1;
            }
            UMGEBUNG_LOG_TRACE("GLFW Initialized.");

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

            m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), NULL, NULL);
            if (m_window == NULL) {
                UMGEBUNG_LOG_CRIT("Failed to create GLFW window");
                glfwTerminate();
                return -1;
            }
            UMGEBUNG_LOG_TRACE("GLFW Window Created.");

            glfwMakeContextCurrent(m_window);
            glfwSwapInterval(1); // Enable vsync

            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                UMGEBUNG_LOG_CRIT("Failed to initialize GLAD");
                return -1;
            }
            UMGEBUNG_LOG_TRACE("GLAD Initialized. OpenGL Version: {}", (const char*)glGetString(GL_VERSION));

            glEnable(GL_DEPTH_TEST);

            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO& io = ImGui::GetIO(); (void)io;
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

            ImGui::StyleColorsDark();
            ImGuiStyle& style = ImGui::GetStyle();
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
                style.WindowRounding = 0.0f;
                style.Colors[ImGuiCol_WindowBg].w = 1.0f;
            }

            ImGui_ImplGlfw_InitForOpenGL(m_window, true);
            ImGui_ImplOpenGL3_Init("#version 460");
            UMGEBUNG_LOG_TRACE("ImGui Initialized.");

            return 0;
        }

        bool Window::shouldClose() const {
            return glfwWindowShouldClose(m_window);
        }

        // NEW function
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
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
        }

        void Window::endImGuiFrame() {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            ImGuiIO& io = ImGui::GetIO();
            if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
                GLFWwindow* backup_current_context = glfwGetCurrentContext();
                ImGui::UpdatePlatformWindows();
                ImGui::RenderPlatformWindowsDefault();
                glfwMakeContextCurrent(backup_current_context);
            }
        }
    }
}