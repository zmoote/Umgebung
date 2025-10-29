#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include "umgebung/scene/Scene.hpp"

// Includes for ALL components
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/ecs/components/Soul.hpp"
#include "umgebung/ecs/components/Consciousness.hpp"

#include <imgui.h>
#include <string>
#include <cstring>
#include <glm/gtc/quaternion.hpp> // For glm::normalize

namespace Umgebung {
    namespace ui {
        namespace imgui {

            // --- Remove the DrawRemoveComponentButton helper function entirely ---
            // template<typename T>
            // void DrawRemoveComponentButton(...) { ... }


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

                        // --- Name Component (Not Removable) ---
                        // Check must happen before potential early return from remove menu
                        bool hasName = registry.all_of<ecs::components::Name>(selectedEntity);
                        if (hasName) {
                            auto& name = registry.get<ecs::components::Name>(selectedEntity);
                            if (ImGui::CollapsingHeader("Name", ImGuiTreeNodeFlags_DefaultOpen)) {
                                // ... InputText for name ...
                                char buffer[256];
                                strncpy_s(buffer, sizeof(buffer), name.name.c_str(), sizeof(buffer) - 1);
                                ImGui::Text("Name");
                                ImGui::SameLine();
                                if (ImGui::InputText("##Name", buffer, sizeof(buffer))) {
                                    name.name = std::string(buffer);
                                }
                            }
                        }

                        // --- Transform Component (Not Removable) ---
                        // Check must happen before potential early return from remove menu
                        bool hasTransform = registry.all_of<ecs::components::Transform>(selectedEntity);
                        if (hasTransform) {
                            auto& transform = registry.get<ecs::components::Transform>(selectedEntity);
                            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                                // Use ImGui::DragFloat3 to edit the glm::vec3 fields
                                // ImGui:DragFloat4 for Quaternion
                                // The "##" hides the label for the widget but keeps the ID unique
                                ImGui::Text("Position");
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Position", &transform.position[0], 0.1f);

                                ImGui::Text("Rotation");
                                ImGui::SameLine();
                                if (ImGui::DragFloat4("##Rotation", &transform.rotation[0], 0.01f)) {
                                    transform.rotation = glm::normalize(transform.rotation);
                                }

                                ImGui::Text("Scale   "); // Added spaces for alignment
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Scale", &transform.scale[0], 0.1f);
                            }
                        }

                        // --- Renderable Component (Removable via Context Menu) ---
                        bool hasRenderable = registry.all_of<ecs::components::Renderable>(selectedEntity);
                        if (hasRenderable) {
                            // Draw header first, store its open state
                            bool open = ImGui::CollapsingHeader("Renderable", ImGuiTreeNodeFlags_DefaultOpen);

                            // Begin context menu attached to the CollapsingHeader
                            if (ImGui::BeginPopupContextItem("RenderableContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##Renderable")) {
                                    registry.remove<ecs::components::Renderable>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    // *** IMPORTANT: Need to end the popup before returning ***
                                    ImGui::EndPopup();
                                    // Return immediately to avoid accessing removed component
                                    ImGui::End(); // End the properties panel window
                                    return;       // Exit the function for this frame
                                }
                                ImGui::EndPopup();
                            }

                            // Only draw fields if component still exists and header is open
                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::Renderable>(selectedEntity) && open) {
                                auto& renderable = registry.get<ecs::components::Renderable>(selectedEntity);
                                ImGui::Text("Mesh Tag: %s", renderable.meshTag.c_str());
                                ImGui::ColorEdit4("Color", &renderable.color[0]);
                                ImGui::Text("Mesh Ptr: %s", (renderable.mesh ? "Assigned" : "None"));
                            }
                        }


                        // --- Soul Component (Removable via Context Menu) ---
                        bool hasSoul = registry.all_of<ecs::components::Soul>(selectedEntity);
                        if (hasSoul) {
                            bool open = ImGui::CollapsingHeader("Soul", ImGuiTreeNodeFlags_DefaultOpen);

                            // Context Menu
                            if (ImGui::BeginPopupContextItem("SoulContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##Soul")) {
                                    registry.remove<ecs::components::Soul>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    ImGui::EndPopup();
                                    ImGui::End(); // End window
                                    return;       // Exit function
                                }
                                ImGui::EndPopup();
                            }

                            // Draw fields if component still exists and header is open
                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::Soul>(selectedEntity) && open) {
                                ImGui::Text("Soul component exists.");
                            }
                        }

                        // --- Consciousness Component (Removable via Context Menu) ---
                        bool hasConsciousness = registry.all_of<ecs::components::Consciousness>(selectedEntity);
                        if (hasConsciousness) {
                            bool open = ImGui::CollapsingHeader("Consciousness", ImGuiTreeNodeFlags_DefaultOpen);

                            // Context Menu
                            if (ImGui::BeginPopupContextItem("ConsciousnessContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##Consciousness")) {
                                    registry.remove<ecs::components::Consciousness>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    ImGui::EndPopup();
                                    ImGui::End(); // End window
                                    return;       // Exit function
                                }
                                ImGui::EndPopup();
                            }

                            // Draw fields if component still exists and header is open
                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::Consciousness>(selectedEntity) && open) {
                                ImGui::Text("Consciousness component exists.");
                            }
                        }


                        // --- Add Component Button & Popup (Remains the same) ---
                        ImGui::Separator();
                        ImGui::Spacing();
                        if (ImGui::Button("Add Component", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                            ImGui::OpenPopup("AddComponentPopup");
                        }

                        if (ImGui::BeginPopup("AddComponentPopup")) {
                            ImGui::Text("Available Components");
                            ImGui::Separator();

                            // Use cached checks from above for slightly better performance
                            if (!hasRenderable) {
                                if (ImGui::MenuItem("Renderable")) {
                                    registry.emplace<ecs::components::Renderable>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                }
                            }
                            if (!hasSoul) {
                                if (ImGui::MenuItem("Soul")) {
                                    registry.emplace<ecs::components::Soul>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                }
                            }
                            if (!hasConsciousness) {
                                if (ImGui::MenuItem("Consciousness")) {
                                    registry.emplace<ecs::components::Consciousness>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                }
                            }
                            // Add future components here
                            ImGui::EndPopup();
                        }
                        // --- End Add Component ---


                        ImGui::Separator(); // Separator before Delete Entity button
                        if (ImGui::Button("Delete Entity")) {
                            scene_->destroyEntity(selectedEntity);
                            // No need to explicitly return early here, selectedEntity will be null next frame
                        }
                    } // End else (entity selected)
                } // End ImGui::Begin
                ImGui::End();
            } // End onUIRender

        } // namespace imgui
    } // namespace ui
} // namespace Umgebung