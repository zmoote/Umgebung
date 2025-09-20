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
                
                if (ImGui::Begin(name_.c_str())) {
                    // Window content goes here

                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}