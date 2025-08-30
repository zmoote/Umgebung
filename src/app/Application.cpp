#include "umgebung/app/Application.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/ui/imgui/ViewportPanel.hpp" // NEW
#include <glm/glm.hpp>

namespace Umgebung::app {

    Application::Application() { UMGEBUNG_LOG_INFO("Application created."); }

    Application::~Application() {
        shutdown();
    }

    int Application::init() {
        m_configManager = std::make_unique<util::ConfigManager>();
        m_configManager->loadConfig("assets/config/CameraLevels.json");

        m_window = std::make_unique<ui::Window>(1600, 900, "Umgebung");
        if (m_window->init() != 0) {
            UMGEBUNG_LOG_CRIT("Window initialization failed.");
            return -1;
        }

        m_framebuffer = std::make_unique<Framebuffer>(1600, 900);
        m_camera = std::make_unique<renderer::Camera>(*m_configManager, 1600.0f, 900.0f);
        m_camera->setCurrentZoomLevel("Planetary");

        m_shader = std::make_unique<renderer::gl::Shader>("assets/shaders/simple.vert", "assets/shaders/simple.frag");

        m_renderer = std::make_unique<renderer::Renderer>();
        m_renderer->init();

        // Add the new ViewportPanel and pass it the framebuffer and camera
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
        UMGEBUNG_LOG_INFO("Application shutdown complete.");
    }

    void Application::run() {
        if (!m_isRunning) return;
        UMGEBUNG_LOG_INFO("Starting main loop...");

        while (!m_window->shouldClose() && m_isRunning) {

            // --- 1. RENDER 3D SCENE TO FRAMEBUFFER ---
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
            // --- END 3D SCENE RENDERING ---


            // --- 2. RENDER UI ---
            m_window->beginFrame();
            m_window->beginImGuiFrame();

            ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

            for (const auto& panel : m_panels) {
                panel->render();
            }

            m_window->endImGuiFrame();
            m_window->endFrame();
        }
    }
}