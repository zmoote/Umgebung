#pragma once

#include "umgebung/ui/Window.hpp"
#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/util/Config.hpp"
#include "umgebung/renderer/Framebuffer.hpp" // FIX: Include the full header
#include <vector>
#include <memory>

namespace Umgebung {
    namespace app {
        class Application {
        public:
            Application();
            ~Application();

            int init();
            void run();
            void shutdown();

        private:
            bool m_isRunning = false;
            std::unique_ptr<ui::Window> m_window;
            std::vector<std::unique_ptr<ui::imgui::Panel>> m_panels;

            std::unique_ptr<util::ConfigManager> m_configManager;
            std::unique_ptr<renderer::Camera> m_camera;
            std::unique_ptr<renderer::Renderer> m_renderer;
            std::unique_ptr<renderer::gl::Shader> m_shader;
            std::unique_ptr<Framebuffer> m_framebuffer;
        };
    }
}