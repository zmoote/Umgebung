#include "umgebung/ui/imgui/ConsolePanel.hpp"
#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            ConsolePanel::ConsolePanel()
                : Panel("Console")
            {

            }

            void ConsolePanel::onUIRender() {

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