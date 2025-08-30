#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include "umgebung/renderer/Camera.hpp"
#include <imgui.h>

namespace umgebung::ui::imgui {
    ViewportPanel::ViewportPanel(renderer::Framebuffer& framebuffer, renderer::Camera& camera)
        : m_framebuffer(framebuffer), m_camera(camera) {
    }

    void ViewportPanel::render() {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("Viewport");

        ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
        uint32_t textureID = m_framebuffer.getColorAttachmentRendererID();

        // FIX: Use a proper cast for the texture ID
        ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(textureID)),
            viewportPanelSize, ImVec2(0, 1), ImVec2(1, 0));

        ImGui::End();
        ImGui::PopStyleVar();
    }
}