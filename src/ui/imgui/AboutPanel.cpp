#include "umgebung/ui/imgui/AboutPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {
    AboutPanel::AboutPanel() : Panel("About") 
    {
        flags_ |= ImGuiWindowFlags_NoResize;
        flags_ |= ImGuiWindowFlags_NoCollapse;
        flags_ |= ImGuiWindowFlags_NoScrollbar;
        flags_ |= ImGuiWindowFlags_NoDocking;
    }

    void AboutPanel::onUIRender() {
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

}