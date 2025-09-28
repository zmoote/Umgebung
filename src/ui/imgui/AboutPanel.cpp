#include "umgebung/ui/imgui/AboutPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

    // It just passes its name to the base Panel constructor
    AboutPanel::AboutPanel() : Panel("About") 
    {
        flags_ |= ImGuiWindowFlags_NoResize;
        flags_ |= ImGuiWindowFlags_NoCollapse;
        flags_ |= ImGuiWindowFlags_NoScrollbar;
        flags_ |= ImGuiWindowFlags_NoDocking;
    }

    void AboutPanel::onUIRender() {

        // Don't render if the panel is closed
        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {
            ImGui::Text("Umgebung");
            ImGui::Text("Version 0.1a");
            ImGui::Separator();
            ImGui::Text("SNHU Physics Capstone");
            ImGui::Text("Copyright (C) Zachary Moote 2025");
        }
        ImGui::End();
    }

} // namespace Umgebung::ui::imgui