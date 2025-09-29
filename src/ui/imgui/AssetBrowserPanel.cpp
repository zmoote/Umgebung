#include "umgebung/ui/imgui/AssetBrowserPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            AssetBrowserPanel::AssetBrowserPanel()
                : Panel("Asset Browser")
            {

            }

            void AssetBrowserPanel::onUIRender() {

                if (!m_isOpen) {
                    return;
                }
                
                if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {

                }
                ImGui::End();
            }

        }
    }
}