#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Name.hpp" // Make sure this is included

#include <imgui.h>
#include <entt/entt.hpp>
#include <string>

namespace Umgebung::ui::imgui {

    HierarchyPanel::HierarchyPanel(scene::Scene* scene)
        : Panel("Hierarchy"), scene_(scene) {
    }

    void HierarchyPanel::onUIRender() {

        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {

            if (ImGui::Button("Create Entity")) {
                entt::entity newEntity = scene_->createEntity();
                scene_->setSelectedEntity(newEntity);
            }
            ImGui::Separator();

            auto& registry = scene_->getRegistry();
            entt::entity currentSelected = scene_->getSelectedEntity();

            auto nameView = registry.view<ecs::components::Name>();

            // --- Start of Fix ---
            for (auto [entity, name] : nameView.each()) {

                const std::string& entityName = name.name;

                // 1. Check if the name is empty and provide a placeholder
                const char* displayName = entityName.empty() ? "(Unnamed Entity)" : entityName.c_str();

                // 2. Create a unique ID for ImGui (e.g., "(Unnamed Entity)##123")
                //    ImGui will only display the part before the "##"
                std::string uniqueLabel = std::string(displayName) + "##" + std::to_string(static_cast<uint32_t>(entity));

                // 3. Pass the safe, unique label to ImGui
                bool isSelected = (currentSelected == entity);
                if (ImGui::Selectable(uniqueLabel.c_str(), isSelected)) {
                    scene_->setSelectedEntity(entity);
                }
            }
            // --- End of Fix ---

            // Deselect if the user clicks in an empty area of the panel
            if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
                scene_->setSelectedEntity(entt::null);
            }

        }
        ImGui::End();
    }

}