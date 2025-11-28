/**
 * @file ViewportPanel.cpp
 * @brief Implements the ViewportPanel class.
 */
#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include "umgebung/app/Application.hpp" // For AppState enum

#include <imgui.h>
#include <imgui_internal.h>

namespace Umgebung::ui::imgui {

ViewportPanel::ViewportPanel(renderer::Framebuffer* framebuffer, std::function<app::AppState()> getAppState)
    : Panel("Viewport", true), framebuffer_(framebuffer), getAppState_(getAppState) {
}

void ViewportPanel::onUIRender() {
    if (!m_isOpen) {
        return;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport", &m_isOpen);
    ImGui::PopStyleVar();

    focused_ = ImGui::IsWindowFocused();

    ImVec2 viewportSize = ImGui::GetContentRegionAvail();
    size_ = { viewportSize.x, viewportSize.y };
    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

    if (viewportSize.x > 0 && viewportSize.y > 0) {
        // Update ViewportPanel's size based on available content region
        ImGui::Image(
            (ImTextureID)(intptr_t)(framebuffer_->getColorAttachmentID()),
            viewportSize,
            ImVec2(0, 1), ImVec2(1, 0)
        );
    }

    // Apply styling based on AppState
    app::AppState state = getAppState_();
    ImVec4 tintColor = ImVec4(0.0f, 0.0f, 0.0f, 0.0f); // Transparent by default

    if (state == app::AppState::Simulate) {
        tintColor = ImVec4(0.0f, 1.0f, 0.0f, 0.1f); // Green tint for simulating
    } else if (state == app::AppState::Paused) {
        tintColor = ImVec4(1.0f, 1.0f, 0.0f, 0.1f); // Yellow tint for paused
    }

    if (tintColor.w > 0.0f) { 
        ImGui::GetWindowDrawList()->AddRectFilled(
            cursorScreenPos, 
            ImVec2(cursorScreenPos.x + viewportSize.x, cursorScreenPos.y + viewportSize.y), 
            ImGui::GetColorU32(tintColor)
        );
    }

    ImGui::End();
}

} // namespace Umgebung::ui::imgui