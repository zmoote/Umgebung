#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            PropertiesPanel::PropertiesPanel(scene::Scene* scene)
                : Panel("Properties"), scene_(scene) // Call the base class constructor with the title
            {
                // You can set specific flags here if you want
                //m_flags = ImGuiWindowFlags_NoResize;
            }

            void PropertiesPanel::onUIRender() {
                
                if (ImGui::Begin(name_.c_str())) {
                    // Window content goes here

                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}