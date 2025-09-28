#include "umgebung/ui/imgui/ConsolePanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            ConsolePanel::ConsolePanel()
                : Panel("Console") // Call the base class constructor with the title
            {
                // You can set specific flags here if you want
                // m_flags = ImGuiWindowFlags_NoResize;
            }

            void ConsolePanel::onUIRender() {

                // Don't render if the panel is closed
                if (!m_isOpen) {
                    return;
                }
                
                if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {
                    // Window content goes here

                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}