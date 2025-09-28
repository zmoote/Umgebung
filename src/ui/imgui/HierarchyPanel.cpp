#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/scene/Scene.hpp"
#include <imgui.h>
#include <entt/entt.hpp> // Make sure to include entt

namespace Umgebung::ui::imgui {

    HierarchyPanel::HierarchyPanel(scene::Scene* scene)
        : Panel("Hierarchy"), scene_(scene) {
    }

    void HierarchyPanel::onUIRender() {

        // Don't render if the panel is closed
        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {
            if (scene_) {
                auto& registry = scene_->getRegistry();

                // --- THIS IS THE FIX ---
                // This is the modern, correct way to iterate over all entities with EnTT.
                for (auto entityID : registry.view<entt::entity>())
                {
                    // You can add more sophisticated drawing here later
                    ImGui::Text("Entity: %u", entt::to_entity(entityID));
                }
            }
        }
        ImGui::End();
    }

} // namespace Umgebung::ui::imgui