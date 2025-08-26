#include "umgebung/app/Application.hpp"
#include "umgebung/util/LogMacros.hpp"
#include <imgui.h> // For the demo window

namespace Umgebung
{
    namespace app
    {
        Application::Application()
        {
            UMGEBUNG_LOG_INFO("Application created.");
        }

        Application::~Application()
        {
            UMGEBUNG_LOG_INFO("Application destroyed.");
        }

        int Application::init()
        {
            // Create the window
            m_window = std::make_unique<ui::Window>(1600, 900, "Umgebung");
            if (m_window->init() != 0)
            {
                UMGEBUNG_LOG_CRIT("Window initialization failed. Shutting down.");
                return -1;
            }

            m_isRunning = true;
            return 0;
        }

        void Application::run()
        {
            if (!m_isRunning) {
                UMGEBUNG_LOG_WARN("Application not initialized. Call init() before run().");
                return;
            }

            UMGEBUNG_LOG_INFO("Starting main loop...");

            while (!m_window->shouldClose())
            {
                m_window->beginFrame();
                m_window->beginImGuiFrame();

                // --- Your application logic and rendering will go here ---
                // For now, let's just show a demo window.
                ImGui::ShowDemoWindow();
                // --------------------------------------------------------

                m_window->endImGuiFrame();
                m_window->endFrame();
            }
            UMGEBUNG_LOG_INFO("Main loop finished.");
        }
    } // namespace app
} // namespace Umgebung