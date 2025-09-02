#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            PropertiesPanel::PropertiesPanel()
                : Panel("Properties") // Call the base class constructor with the title
            {
                // You can set specific flags here if you want
                //m_flags = ImGuiWindowFlags_NoResize;
            }

            void PropertiesPanel::render() {
                // Don't render if the panel is closed
                if (!m_isOpen) {
                    return;
                }

                // Begin the ImGui window. The `&m_isOpen` parameter adds a close
                // button that will automatically update our boolean.
                if (ImGui::Begin(m_title.c_str(), &m_isOpen, m_flags)) {
                    // Window content goes here

                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}