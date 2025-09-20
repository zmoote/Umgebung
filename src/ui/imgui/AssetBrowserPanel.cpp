#include "umgebung/ui/imgui/AssetBrowserPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            AssetBrowserPanel::AssetBrowserPanel()
                : Panel("Asset Browser") // Call the base class constructor with the title
            {
                // You can set specific flags here if you want
                // m_flags = ImGuiWindowFlags_NoResize;
            }

            void AssetBrowserPanel::onUIRender() {
                
                if (ImGui::Begin(name_.c_str())) {
                    // Window content goes here

                }
                // End the ImGui window
                ImGui::End();
            }

        }
    }
}