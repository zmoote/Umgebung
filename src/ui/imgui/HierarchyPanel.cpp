#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/scene/Scene.hpp"
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

            // Iterate over all entities
            registry.view<entt::entity>().each([&](auto entity) {
                // TODO: We will replace this with a proper "Name" component later
                std::string entityName = "Entity " + std::to_string(static_cast<uint32_t>(entity));

                // ImGui::Selectable returns true if clicked
                bool isSelected = (currentSelected == entity);
                if (ImGui::Selectable(entityName.c_str(), isSelected)) {
                    // If clicked, update the scene's selected entity
                    scene_->setSelectedEntity(entity);
                }
                });

            // Deselect if the user clicks in an empty area of the panel
            if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
                scene_->setSelectedEntity(entt::null);
            }
            
        }
        ImGui::End();
    }

}