#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include "umgebung/scene/Scene.hpp"

#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Soul.hpp"
#include "umgebung/ecs/components/Consciousness.hpp"

#include <imgui.h>

namespace Umgebung {
    namespace ui {
        namespace imgui {

            PropertiesPanel::PropertiesPanel(scene::Scene* scene)
                : Panel("Properties"), scene_(scene)
            {

            }

            void PropertiesPanel::onUIRender() {

                if (!m_isOpen) {
                    return;
                }
                
                if (ImGui::Begin(name_.c_str(), &m_isOpen, flags_)) {

                    entt::entity selectedEntity = scene_->getSelectedEntity();
                    auto& registry = scene_->getRegistry();

                    if (selectedEntity == entt::null) {
                        ImGui::Text("No entity selected.");
                    }
                    else {

                        // --- Transform Component ---
                        // Check if the entity has this component
                        if (registry.all_of<ecs::components::TransformComponent>(selectedEntity)) {
                            // Get a reference to the component
                            auto& transform = registry.get<ecs::components::TransformComponent>(selectedEntity);

                            // ImGuiTreeNodeFlags_DefaultOpen makes the header open by default
                            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                                // Use ImGui::DragFloat3 to edit the glm::vec3 fields
                                // The "##" hides the label for the widget but keeps the ID unique
                                ImGui::Text("Position");
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Position", &transform.position[0], 0.1f);

                                ImGui::Text("Rotation");
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Rotation", &transform.rotation[0], 0.1f);

                                ImGui::Text("Scale   "); // Added spaces for alignment
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Scale", &transform.scale[0], 0.1f);
                            }
                        }

                        // --- Renderable Component ---
                        if (registry.all_of<ecs::components::RenderableComponent>(selectedEntity)) {
                            auto& renderable = registry.get<ecs::components::RenderableComponent>(selectedEntity);
                            if (ImGui::CollapsingHeader("Renderable", ImGuiTreeNodeFlags_DefaultOpen)) {
                                // We can't easily edit the mesh pointer, but we can show info
                                ImGui::Text("Mesh: %s", (renderable.mesh ? "Assigned" : "None"));
                                // You could add color, material properties, etc. here later
                            }
                        }

                        // --- Soul Component (Example) ---
                        if (registry.all_of<ecs::components::Soul>(selectedEntity)) {
                            if (ImGui::CollapsingHeader("Soul", ImGuiTreeNodeFlags_DefaultOpen)) {
                                ImGui::Text("Soul component exists.");
                                // Add widgets for Soul properties here
                            }
                        }

                        // --- Consciousness Component (Example) ---
                        if (registry.all_of<ecs::components::Consciousness>(selectedEntity)) {
                            if (ImGui::CollapsingHeader("Consciousness", ImGuiTreeNodeFlags_DefaultOpen)) {
                                ImGui::Text("Consciousness component exists.");
                                // Add widgets for Consciousness properties here
                            }
                        }

                        ImGui::Separator();
                        if (ImGui::Button("Delete Entity")) {
                            scene_->destroyEntity(selectedEntity);

                            ImGui::End();
                            return;
                        }
                    }
                }
                ImGui::End();
            }

        }
    }
}