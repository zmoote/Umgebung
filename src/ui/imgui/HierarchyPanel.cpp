/**
 * @file HierarchyPanel.cpp
 * @brief Implements the HierarchyPanel class.
 */
#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Name.hpp"

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

            for (auto [entity, name] : nameView.each()) {

                const std::string& entityName = name.name;

                const char* displayName = entityName.empty() ? "(Unnamed Entity)" : entityName.c_str();

                std::string uniqueLabel = std::string(displayName) + "##" + std::to_string(static_cast<uint32_t>(entity));

                bool isSelected = (currentSelected == entity);
                if (ImGui::Selectable(uniqueLabel.c_str(), isSelected)) {
                    scene_->setSelectedEntity(entity);
                }
            }

            if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
                scene_->setSelectedEntity(entt::null);
            }

        }
        ImGui::End();
    }

}