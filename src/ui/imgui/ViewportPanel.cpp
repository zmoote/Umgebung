/**
 * @file ViewportPanel.cpp
 * @brief Implements the ViewportPanel class.
 */
#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {
    ViewportPanel::ViewportPanel(renderer::Framebuffer* framebuffer)
        : Panel("Viewport", false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse),
        framebuffer_(framebuffer) {
    }

    void ViewportPanel::onUIRender() {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
        ImGui::Begin(name_.c_str(), nullptr, flags_);

        focused_ = ImGui::IsWindowFocused() && ImGui::IsWindowHovered();

        ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
        size_ = { viewportPanelSize.x, viewportPanelSize.y };

        uint32_t textureID = framebuffer_->getColorAttachmentID();

        ImGui::Image((ImTextureID)textureID, viewportPanelSize, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });

        ImGui::End();
        ImGui::PopStyleVar();
    }

}