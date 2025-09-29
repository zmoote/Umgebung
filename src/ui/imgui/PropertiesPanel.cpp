#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            PropertiesPanel::PropertiesPanel(scene::Scene* scene)
                : Panel("Properties"), scene_(scene)
            {

            }

            void PropertiesPanel::onUIRender() {

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