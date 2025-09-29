#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/scene/Scene.hpp"
#include <imgui.h>
#include <entt/entt.hpp>

namespace Umgebung::ui::imgui {

    HierarchyPanel::HierarchyPanel(scene::Scene* scene)
        : Panel("Hierarchy"), scene_(scene) {
    }

    void HierarchyPanel::onUIRender() {

        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {
            if (scene_) {
                auto& registry = scene_->getRegistry();

                for (auto entityID : registry.view<entt::entity>())
                {
                    ImGui::Text("Entity: %u", entt::to_entity(entityID));
                }
            }
        }
        ImGui::End();
    }

}