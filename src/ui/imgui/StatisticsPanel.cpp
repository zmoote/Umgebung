#include "umgebung/ui/imgui/StatisticsPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            StatisticsPanel::StatisticsPanel()
                : Panel("Statistics")
            {
                flags_ |= ImGuiWindowFlags_NoResize;
                flags_ |= ImGuiWindowFlags_NoCollapse;
                flags_ |= ImGuiWindowFlags_NoDocking;
                flags_ |= ImGuiWindowFlags_NoScrollbar;
            }

            void StatisticsPanel::onUIRender() {

                if (!m_isOpen) {
                    return;
                }
                
                if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {
                    ImGui::Text("Frame Rate:");
                    ImGui::SameLine();
                    ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
                }
                ImGui::End();
            }

        }
    }
}