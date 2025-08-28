#include "umgebung/ui/imgui/StatisticsPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            StatisticsPanel::StatisticsPanel()
                : Panel("Statistics") // Call the base class constructor with the title
            {
                // You can set specific flags here if you want
                m_flags |= ImGuiWindowFlags_NoResize;
                m_flags |= ImGuiWindowFlags_NoCollapse;
                m_flags |= ImGuiWindowFlags_NoDocking;
                m_flags |= ImGuiWindowFlags_NoScrollbar;
            }

            void StatisticsPanel::render() {
                // Don't render if the panel is closed
                if (!m_isOpen) {
                    return;
                }

                // Begin the ImGui window. The `&m_isOpen` parameter adds a close
                // button that will automatically update our boolean.
                if (ImGui::Begin(m_title.c_str(), &m_isOpen, m_flags)) {
                    // Window content goes here
                    ImGui::Text("Frame Rate:");
                    ImGui::SameLine();
                    ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}