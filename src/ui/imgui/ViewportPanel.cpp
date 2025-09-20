#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

    ViewportPanel::ViewportPanel(renderer::Framebuffer* framebuffer)
        : Panel("Viewport"), framebuffer_(framebuffer) {
    }

    void ViewportPanel::onUIRender() {
        // Remove padding from the window
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
        ImGui::Begin(name_.c_str());

        // Get the size of the content region
        ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
        size_ = { viewportPanelSize.x, viewportPanelSize.y };

        // Get the texture ID from our framebuffer
        uint32_t textureID = framebuffer_->getColorAttachmentID();

        // --- THIS IS THE FIX ---
        // Let's use a direct cast to the ImTextureID type.
        ImGui::Image((ImTextureID)textureID, viewportPanelSize, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });

        ImGui::End();
        ImGui::PopStyleVar();
    }

} // namespace Umgebung::ui::imgui