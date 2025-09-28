#include "umgebung/ui/imgui/AboutPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

    // It just passes its name to the base Panel constructor
    AboutPanel::AboutPanel() : Panel("About", ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize) {}

    void AboutPanel::onUIRender() {
        if (ImGui::Begin(name_.c_str())) {
            ImGui::Text("Umgebung");
            ImGui::Text("Version 0.1a");
            ImGui::Separator();
            ImGui::Text("SNHU Physics Capstone");
            ImGui::Text("Copyright (C) Zachary Moote 2025");
        }
        ImGui::End();
    }

} // namespace Umgebung::ui::imgui