#include "umgebung/ui/imgui/AboutPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            AboutPanel::AboutPanel()
                : Panel("About Umgebung") // Call the base class constructor with the title
            {
                // You can set specific flags here if you want
                m_flags |= ImGuiWindowFlags_NoResize;
                m_flags |= ImGuiWindowFlags_NoCollapse;
                m_flags |= ImGuiWindowFlags_NoScrollbar;
                m_flags |= ImGuiWindowFlags_NoDocking;
            }

            void AboutPanel::render() {
                // Don't render if the panel is closed
                if (!m_isOpen) {
                    return;
                }

                // Begin the ImGui window. The `&m_isOpen` parameter adds a close
                // button that will automatically update our boolean.
                if (ImGui::Begin(m_title.c_str(), &m_isOpen, m_flags)) {

                    // Get current cursor position to calculate relative offsets
                    ImVec2 currentCursorPos = ImGui::GetCursorPos();

                    // Position a button at a specific absolute position within the panel
                    ImGui::SetCursorPos(ImVec2(641, 529)); // Max_X=764, Max_Y=574

                    // Window content goes here
                    if (ImGui::Button("OK", ImVec2(100,25))) 
                    {
                        m_isOpen = false;
                    }
                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}