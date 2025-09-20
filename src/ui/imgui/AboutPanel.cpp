#include "umgebung/ui/imgui/AboutPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

    // It just passes its name to the base Panel constructor
    AboutPanel::AboutPanel() : Panel("About") {}

    void AboutPanel::onUIRender() {
        if (ImGui::Begin(name_.c_str())) {
            ImGui::Text("Umgebung Engine");
            ImGui::Text("Version 0.1a");
            ImGui::Separator();
            ImGui::Text("A personal project.");
        }
        ImGui::End();
    }

} // namespace Umgebung::ui::imgui