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