/**
 * @file AboutPanel.cpp
 * @brief Implementation of the floating non-docking `"About"` utility panel
 *        adapted directly from `Umgebung-legacy`.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/AboutPanel.hpp"
#include "umgebung/ui/UIContext.hpp"

#include <imgui.h>

namespace umgebung::ui {

    AboutPanel::AboutPanel()
        : Panel(
              "About",
              false,
              ImGuiWindowFlags_NoResize |
                  ImGuiWindowFlags_NoCollapse |
                  ImGuiWindowFlags_NoScrollbar |
                  ImGuiWindowFlags_NoDocking |
                  ImGuiWindowFlags_AlwaysAutoResize
          ) {}

    void AboutPanel::OnUIRender(UIContext& ctx) {
        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(m_name.c_str(), &m_isOpen, m_flags)) {
            m_focused = ImGui::IsWindowFocused();
            m_hovered = ImGui::IsWindowHovered();

            ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "Umgebung");
            ImGui::Text("Version %s (C++20 / CUDA / Vulkan + NVRHI)", ctx.engineVersion ? ctx.engineVersion : "0.0.1");
            ImGui::Separator();
            ImGui::Text("Multi-Scale Reality Simulation Engine & Computational Physics Testbed");
            ImGui::Text("Originated as SNHU Physics Capstone");
            ImGui::Text("Copyright (C) Zachary Moote 2026");

            if (!ctx.selectedVulkanGpuName.empty()) {
                ImGui::Separator();
                ImGui::TextDisabled("Active GPU: %s", ctx.selectedVulkanGpuName.c_str());
            }
        }
        ImGui::End();
    }

} // namespace umgebung::ui
