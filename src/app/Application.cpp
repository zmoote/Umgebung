#include "umgebung/app/Application.hpp"
#include "umgebung/util/LogMacros.hpp"

// Include any concrete panels you create
#include "umgebung/ui/imgui/StatisticsPanel.hpp"
#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/ui/imgui/AboutPanel.hpp"

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

            // --- Create and add panels here ---
            m_panels.push_back(std::make_unique<ui::imgui::HierarchyPanel>());
            
            // m_panels.push_back(std::make_unique<ui::imgui::ExamplePanel>());
            // ------------------------------------

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

            while (!m_window->shouldClose() && m_isRunning)
            {
                m_window->beginFrame();
                m_window->beginImGuiFrame();

                // ----------------- NEW DOCKSPACE CODE (START) -----------------

               // Set up the flags for our main window that will host the dockspace.
               // We want it to cover the entire viewport and act as a background.
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
                window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
                window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

                // Set the position and size of this host window to match the main viewport.
                const ImGuiViewport* viewport = ImGui::GetMainViewport();
                ImGui::SetNextWindowPos(viewport->WorkPos);
                ImGui::SetNextWindowSize(viewport->WorkSize);
                ImGui::SetNextWindowViewport(viewport->ID);

                // Remove padding and border for a seamless look.
                ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

                // Begin the host window.
                ImGui::Begin("DockSpace Host", nullptr, window_flags);

                // Pop the style variables.
                ImGui::PopStyleVar(3);

                // Create the actual dockspace.
                ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
                ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

                // Example of a main menu bar.
                if (ImGui::BeginMenuBar())
                {
                    if (ImGui::BeginMenu("File"))
                    {
                        if (ImGui::MenuItem("Exit")) {
                            m_isRunning = false; // Set the flag to exit the loop
                        }
                        ImGui::EndMenu();
                    }
                    
                    ImGui::Spacing();

                    if (ImGui::BeginMenu("Edit")) {
                        ImGui::EndMenu();
                    }
                    
                    ImGui::Spacing();

                    if (ImGui::BeginMenu("View"))
                    {
                        if (ImGui::MenuItem("Statistics"))
                        {
                            m_panels.push_back(std::make_unique<ui::imgui::StatisticsPanel>());
                        }
                        ImGui::EndMenu();
                    }

                    ImGui::Spacing();
                    
                    if (ImGui::BeginMenu("Tools")) {
                        ImGui::EndMenu();
                    }

                    ImGui::Spacing();

                    if (ImGui::BeginMenu("Help")) {
                        if (ImGui::MenuItem("About Umgebung"))
                        {
                            m_panels.push_back(std::make_unique<ui::imgui::AboutPanel>());
                        }
                        ImGui::EndMenu();
                    }

                    ImGui::Spacing();

                    ImGui::EndMenuBar();
                }

                // ----------------- NEW DOCKSPACE CODE (END) -----------------


                // --- Your application logic and rendering will go here ---
                // For now, let's just show a demo window.
                //ImGui::ShowDemoWindow();
                // --------------------------------------------------------

                // --- Render all the UI panels ---
                for (const auto& panel : m_panels)
                {
                    panel->render();
                }
                // --------------------------------

                 // This call ends the host window that contains the dockspace.
                // It's important that it's called AFTER the panels that will live inside it.
                ImGui::End();

                m_window->endImGuiFrame();
                m_window->endFrame();
            }
            UMGEBUNG_LOG_INFO("Main loop finished.");
        }
    } // namespace app
} // namespace Umgebung