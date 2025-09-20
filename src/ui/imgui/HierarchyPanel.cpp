#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/scene/Scene.hpp"
#include <imgui.h>
#include <entt/entt.hpp> // Make sure to include entt

namespace Umgebung::ui::imgui {

    HierarchyPanel::HierarchyPanel(scene::Scene* scene)
        : Panel("Hierarchy"), scene_(scene) {
    }

    void HierarchyPanel::onUIRender() {
        if (ImGui::Begin(name_.c_str())) {
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