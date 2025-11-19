/**
 * @file PropertiesPanel.cpp
 * @brief Implements the PropertiesPanel class.
 */
#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include "umgebung/ui/imgui/FilePickerPanel.hpp"
#include "umgebung/scene/Scene.hpp"

// Includes for ALL components
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/ecs/components/Soul.hpp"
#include "umgebung/ecs/components/Consciousness.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"

#include <imgui.h>
#include <string>
#include <cstring>
#include <glm/gtc/quaternion.hpp> // For glm::normalize

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

                        bool hasName = registry.all_of<ecs::components::Name>(selectedEntity);
                        if (hasName) {
                            auto& name = registry.get<ecs::components::Name>(selectedEntity);
                            if (ImGui::CollapsingHeader("Name", ImGuiTreeNodeFlags_DefaultOpen)) {
                                char buffer[256];
                                strncpy_s(buffer, sizeof(buffer), name.name.c_str(), name.name.length());
                                ImGui::Text("Name");
                                ImGui::SameLine();
                                if (ImGui::InputText("##Name", buffer, sizeof(buffer))) {
                                    name.name = std::string(buffer);
                                }
                            }
                        }

                        bool hasTransform = registry.all_of<ecs::components::Transform>(selectedEntity);
                        if (hasTransform) {
                            auto& transform = registry.get<ecs::components::Transform>(selectedEntity);
                            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                                ImGui::Text("Position");
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Position", &transform.position[0], 0.1f);

                                ImGui::Text("Rotation");
                                ImGui::SameLine();
                                if (ImGui::DragFloat4("##Rotation", &transform.rotation[0], 0.01f)) {
                                    transform.rotation = glm::normalize(transform.rotation);
                                }

                                ImGui::Text("Scale   ");
                                ImGui::SameLine();
                                ImGui::DragFloat3("##Scale", &transform.scale[0], 0.1f);
                            }
                        }

                        bool hasRenderable = registry.all_of<ecs::components::Renderable>(selectedEntity);
                        if (hasRenderable) {
                            bool open = ImGui::CollapsingHeader("Renderable", ImGuiTreeNodeFlags_DefaultOpen);

                            if (ImGui::BeginPopupContextItem("RenderableContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##Renderable")) {
                                    registry.remove<ecs::components::Renderable>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    ImGui::EndPopup();
                                    ImGui::End();
                                    return;
                                }
                                ImGui::EndPopup();
                            }

                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::Renderable>(selectedEntity) && open) {
                                auto& renderable = registry.get<ecs::components::Renderable>(selectedEntity);
                                ImGui::Text("Mesh Tag: %s", renderable.meshTag.c_str());
                                ImGui::SameLine();
                                if (ImGui::Button("...")) {
                                    filePicker_ = std::make_unique<FilePickerPanel>("Select Mesh", "assets/models", [this, selectedEntity](const std::filesystem::path& path) {
                                        auto& registry = scene_->getRegistry();
                                        auto& renderable = registry.get<ecs::components::Renderable>(selectedEntity);
                                        renderable.meshTag = path.generic_string();
                                    });
                                    filePicker_->open();
                                }
                                ImGui::ColorEdit4("Color", &renderable.color[0]);
                                ImGui::Text("Mesh Ptr: %s", (renderable.mesh ? "Assigned" : "None"));
                            }
                        }

                        bool hasRigidBody = registry.all_of<ecs::components::RigidBody>(selectedEntity);
                        if (hasRigidBody) {
                            bool open = ImGui::CollapsingHeader("RigidBody", ImGuiTreeNodeFlags_DefaultOpen);

                            if (ImGui::BeginPopupContextItem("RigidBodyContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##RigidBody")) {
                                    registry.remove<ecs::components::RigidBody>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    ImGui::EndPopup();
                                    ImGui::End();
                                    return;
                                }
                                ImGui::EndPopup();
                            }

                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::RigidBody>(selectedEntity) && open) {
                                auto& rigidBody = registry.get<ecs::components::RigidBody>(selectedEntity);

                                ImGui::Text("Mass");
                                ImGui::SameLine();
                                ImGui::DragFloat("##Mass", &rigidBody.mass, 0.1f, 0.0f);

                                ImGui::Text("Body Type");
                                ImGui::SameLine();
                                const char* bodyTypes[] = { "Static", "Dynamic" };
                                int currentBodyType = static_cast<int>(rigidBody.type);
                                if (ImGui::Combo("##BodyType", &currentBodyType, bodyTypes, IM_ARRAYSIZE(bodyTypes))) {
                                    rigidBody.type = static_cast<ecs::components::RigidBody::BodyType>(currentBodyType);
                                }
                            }
                        }

                        bool hasSoul = registry.all_of<ecs::components::Soul>(selectedEntity);
                        if (hasSoul) {
                            bool open = ImGui::CollapsingHeader("Soul", ImGuiTreeNodeFlags_DefaultOpen);

                            if (ImGui::BeginPopupContextItem("SoulContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##Soul")) {
                                    registry.remove<ecs::components::Soul>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    ImGui::EndPopup();
                                    ImGui::End();
                                    return;
                                }
                                ImGui::EndPopup();
                            }

                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::Soul>(selectedEntity) && open) {
                                ImGui::Text("Soul component exists.");
                            }
                        }

                        bool hasConsciousness = registry.all_of<ecs::components::Consciousness>(selectedEntity);
                        if (hasConsciousness) {
                            bool open = ImGui::CollapsingHeader("Consciousness", ImGuiTreeNodeFlags_DefaultOpen);

                            if (ImGui::BeginPopupContextItem("ConsciousnessContextMenu")) {
                                if (ImGui::MenuItem("Remove Component##Consciousness")) {
                                    registry.remove<ecs::components::Consciousness>(selectedEntity);
                                    ImGui::CloseCurrentPopup();
                                    ImGui::EndPopup();
                                    ImGui::End();
                                    return;
                                }
                                ImGui::EndPopup();
                            }

                            if (registry.valid(selectedEntity) && registry.all_of<ecs::components::Consciousness>(selectedEntity) && open) {
                                ImGui::Text("Consciousness component exists.");
                            }
                        }

                        ImGui::Separator();
                        ImGui::Spacing();
                        if (ImGui::Button("Add Component", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                            ImGui::OpenPopup("AddComponentPopup");
                        }

                        if (ImGui::BeginPopup("AddComponentPopup")) {
                            ImGui::Text("Available Components");
                            ImGui::Separator();

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
                            if (!hasRigidBody) {
                                if (ImGui::BeginMenu("RigidBody")) {
                                    if (ImGui::MenuItem("Static")) {
                                        registry.emplace<ecs::components::RigidBody>(selectedEntity);
                                        ImGui::CloseCurrentPopup();
                                    }
                                    if (ImGui::MenuItem("Dynamic")) {
                                        auto& rb = registry.emplace<ecs::components::RigidBody>(selectedEntity);
                                        rb.type = ecs::components::RigidBody::BodyType::Dynamic;
                                        ImGui::CloseCurrentPopup();
                                    }
                                    ImGui::EndMenu();
                                }
                            }
                            ImGui::EndPopup();
                        }

                        ImGui::Separator();
                        if (ImGui::Button("Delete Entity")) {
                            scene_->destroyEntity(selectedEntity);
                        }
                    }
                }
                if (filePicker_) {
                    filePicker_->onUIRender();
                }
                ImGui::End();
            }

        }
    }
}