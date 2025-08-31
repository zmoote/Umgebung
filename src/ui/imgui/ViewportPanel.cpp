#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include "umgebung/renderer/Camera.hpp"
#include <imgui.h>

namespace Umgebung::ui::imgui {

    ViewportPanel::ViewportPanel(renderer::Framebuffer& framebuffer, renderer::Camera& camera)
        : Panel("Viewport"), m_framebuffer(framebuffer), m_camera(camera) {
    } // FIX: Call base class constructor

    void ViewportPanel::render() {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0.0f, 0.0f });
        ImGui::Begin(m_title.c_str()); // FIX: Use m_title from base Panel class

        ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
        m_framebuffer.resize(static_cast<uint32_t>(viewportPanelSize.x), static_cast<uint32_t>(viewportPanelSize.y));

        uint32_t textureID = m_framebuffer.getColorAttachmentRendererID();

        // FIX: Use a C-style cast, which is the idiomatic way for ImGui's texture ID
        ImGui::Image((ImTextureID)(intptr_t)textureID, viewportPanelSize, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });

        ImGui::End();
        ImGui::PopStyleVar();
    }
}