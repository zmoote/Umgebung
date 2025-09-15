#pragma once

#include "umgebung/ui/Window.hpp"
#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/util/Config.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
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

            // This single helper can replace both panelExists() and panelIsOpen()
            template<typename T>
            T* getPanel() {
                for (const auto& panel : m_panels) {
                    if (T* p = dynamic_cast<T*>(panel.get())) {
                        return p; // Return a pointer to the found panel
                    }
                }
                return nullptr; // Return null if not found
            }

            bool m_isRunning = false;
            std::unique_ptr<ui::Window> m_window;
            std::vector<std::unique_ptr<ui::imgui::Panel>> m_panels;

            std::unique_ptr<util::ConfigManager> m_configManager;
            std::unique_ptr<renderer::Camera> m_camera;
            std::unique_ptr<renderer::Renderer> m_renderer;
            std::unique_ptr<renderer::gl::Shader> m_shader;
            std::unique_ptr<renderer::Framebuffer> m_framebuffer;
        };
    }
}