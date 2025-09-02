#include "umgebung/app/Application.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/ui/imgui/StatisticsPanel.hpp"
#include "umgebung/ui/imgui/AboutPanel.hpp"
#include <glm/glm.hpp>
#include <imgui.h>

namespace Umgebung::app {

    Application::Application() { /* empty */ }
    Application::~Application() { shutdown(); }

    int Application::init() {
        m_configManager = std::make_unique<util::ConfigManager>();
        m_configManager->loadConfig("assets/config/CameraLevels.json");

        m_window = std::make_unique<ui::Window>(1600, 900, "Umgebung");
        if (m_window->init() != 0) {
            UMGEBUNG_LOG_CRIT("Window initialization failed.");
            return -1;
        }

        m_framebuffer = std::make_unique<renderer::Framebuffer>(1600, 900);
        m_camera = std::make_unique<renderer::Camera>(*m_configManager, 1600.0f, 900.0f);
        m_camera->setCurrentZoomLevel("Planetary");

        m_shader = std::make_unique<renderer::gl::Shader>("assets/shaders/simple.vert", "assets/shaders/simple.frag");

        m_renderer = std::make_unique<renderer::Renderer>();
        m_renderer->init();

        m_panels.push_back(std::make_unique<ui::imgui::ViewportPanel>(*m_framebuffer, *m_camera));
        m_panels.push_back(std::make_unique<ui::imgui::HierarchyPanel>());

        m_isRunning = true;
        return 0;
    }

    void Application::shutdown() {
        m_panels.clear();
        m_shader.reset();
        m_renderer.reset();
        m_camera.reset();
        m_framebuffer.reset();
        m_configManager.reset();
        m_window.reset();
    }

    void Application::run() {
        if (!m_isRunning) return;
        UMGEBUNG_LOG_INFO("Starting main loop...");

        while (!m_window->shouldClose() && m_isRunning) {

            // --- 1. RENDER 3D SCENE TO OUR OFF-SCREEN FRAMEBUFFER ---
            m_framebuffer->bind();
            m_window->clear();

            m_shader->use();
            glm::mat4 model = glm::mat4(1.0f);
            glm::mat4 view = m_camera->getViewMatrix();
            glm::mat4 projection = m_camera->getProjectionMatrix();
            m_shader->setMat4("model", model);
            m_shader->setMat4("view", view);
            m_shader->setMat4("projection", projection);
            m_renderer->draw();

            m_framebuffer->unbind();

            // --- 2. RENDER THE UI TO THE MAIN WINDOW ---
            m_window->beginFrame();
            m_window->beginImGuiFrame();

            // --- HOST WINDOW & DOCKSPACE SETUP ---
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->Pos);
            ImGui::SetNextWindowSize(viewport->Size);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("DockSpace Demo", nullptr, window_flags);
            ImGui::PopStyleVar(3);

            // FIX: Use the correct ImGui::DockSpace call
            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

            // --- MAIN MENU BAR ---
            if (ImGui::BeginMainMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Exit")) { m_isRunning = false; }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Tools")) {
                    if (ImGui::MenuItem("Statistics")) { m_panels.push_back(std::make_unique<ui::imgui::StatisticsPanel>()); }
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Help")) {
                    if (ImGui::MenuItem("About Umgebung")) { m_panels.push_back(std::make_unique<ui::imgui::AboutPanel>()); }
                    ImGui::EndMenu();
                }
                
                ImGui::EndMainMenuBar();
            }

            ImGui::End(); // End the host window

            // --- RENDER ALL UI PANELS ---
            for (const auto& panel : m_panels) {
                panel->render();
            }

            m_window->endImGuiFrame();
            m_window->endFrame();
        }
    }
}