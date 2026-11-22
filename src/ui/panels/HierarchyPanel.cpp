/**
 * @file HierarchyPanel.cpp
 * @brief Implementation of the left-docked `"Hierarchy"` panel adapting legacy
 *        Umgebung's `Create...` entity spawner and scale-grouped selectable
 *        entity tree to current Umgebung's three-scope workspace.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/HierarchyPanel.hpp"
#include "umgebung/scene/PhysicsProfile.hpp"
#include "umgebung/ui/UIContext.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace umgebung::ui {

    namespace {

        /**
         * @brief Constructs a pre-configured `scene::BodyEntity` with a unique ID
         *        suitable for insertion into the active scene from the `Create...` menu.
         */
        [[nodiscard]] scene::BodyEntity MakeSpawnBody(
            const scene::UniverseScene& activeScene,
            const std::string& baseName,
            const std::string& category,
            scene::ScaleTier tier,
            double mass,
            double radius,
            double density = 3.0
        ) {
            scene::BodyEntity entity{};
            const std::size_t suffix = activeScene.bodies.size() + 1U;
            entity.id = category + "_" + std::to_string(suffix);
            entity.name = baseName + " " + std::to_string(suffix);
            entity.category = category;
            entity.scaleTier = tier;
            entity.mass = mass;
            entity.radius = radius;
            // Offset newly spawned entities slightly along +X so they do not
            // coincide with an existing central mass at the origin (avoiding 1/r singularity).
            entity.position = {1.25 + 0.35 * static_cast<double>(suffix), 0.0, 0.0};
            entity.velocity = {0.0, 0.65, 0.0};
            entity.vibrationalDensity = density;
            return entity;
        }

    } // anonymous namespace

    HierarchyPanel::HierarchyPanel()
        : Panel("Hierarchy", true) {}

    void HierarchyPanel::RenderCreateEntityPopup(UIContext& ctx) {
        if (!ctx.hasActiveScene || ctx.activeScene == nullptr) {
            return;
        }

        if (ImGui::BeginPopup("CreateEntityPopup")) {
            const auto& sceneRef = *ctx.activeScene;
            auto spawn = [&](scene::BodyEntity entity) {
                if (ctx.onCreateBodyEntity) {
                    ctx.onCreateBodyEntity(entity);
                }
            };

            if (ImGui::BeginMenu("Quantum / Micro")) {
                if (ImGui::MenuItem("Subatomic PSU")) {
                    spawn(MakeSpawnBody(sceneRef, "Planck Spherical Unit", "SubatomicPSU", scene::ScaleTier::SubatomicPSU, 1.0e-3, 0.04, 1.5));
                }
                if (ImGui::MenuItem("Atom")) {
                    spawn(MakeSpawnBody(sceneRef, "Atom", "Atom", scene::ScaleTier::Atom, 1.0e-2, 0.05, 2.0));
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Entity / Device")) {
                if (ImGui::MenuItem("Phryll Crystal (Vogel)")) {
                    spawn(MakeSpawnBody(sceneRef, "Phryll Crystal", "PhryllCrystal", scene::ScaleTier::EntityOrDevice, 1.0, 0.35, 5.0));
                }
                if (ImGui::MenuItem("HEEMFG Cavity (Pais)")) {
                    spawn(MakeSpawnBody(sceneRef, "HEEMFG Cavity", "HEEMFG_Cavity", scene::ScaleTier::EntityOrDevice, 1.0, 0.45, 4.0));
                }
                if (ImGui::MenuItem("Conscious Entity")) {
                    spawn(MakeSpawnBody(sceneRef, "Conscious Entity", "ConsciousEntity", scene::ScaleTier::EntityOrDevice, 1.0, 0.15, 5.5));
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Macro")) {
                if (ImGui::MenuItem("Planet")) {
                    spawn(MakeSpawnBody(sceneRef, "Planet", "Planet", scene::ScaleTier::Planet, 1.0e-3, 0.12, 3.0));
                }
                if (ImGui::MenuItem("Star")) {
                    spawn(MakeSpawnBody(sceneRef, "Star", "Star", scene::ScaleTier::StarSystem, 1.0, 0.28, 4.0));
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Cosmic")) {
                if (ImGui::MenuItem("Galactic Core")) {
                    spawn(MakeSpawnBody(sceneRef, "Galactic Core", "GalacticCore", scene::ScaleTier::Galaxy, 10.0, 0.40, 9.0));
                }
                if (ImGui::MenuItem("Bubble Universe Anchor")) {
                    spawn(MakeSpawnBody(sceneRef, "Bubble Anchor", "UniverseAnchor", scene::ScaleTier::BubbleUniverse, 100.0, 0.55, 12.0));
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();
            if (ImGui::MenuItem("Empty Body")) {
                spawn(MakeSpawnBody(sceneRef, "Body", "Body", scene::ScaleTier::StarSystem, 0.1, 0.08, 3.0));
            }

            ImGui::EndPopup();
        }
    }

    void HierarchyPanel::RenderSubSimulationsForParent(
        UIContext& ctx,
        const scene::UniverseScene& sceneRef,
        const std::string& parentId
    ) {
        for (std::size_t sIdx = 0U; sIdx < sceneRef.subSimulations.size(); ++sIdx) {
            const auto& sub = sceneRef.subSimulations[sIdx];
            if (sub.parentId != parentId) {
                continue;
            }

            bool hasNestedSubs = false;
            for (const auto& cand : sceneRef.subSimulations) {
                if (cand.parentId == sub.id) {
                    hasNestedSubs = true;
                    break;
                }
            }

            const std::string nodeLabel =
                sub.name + " [" + scene::ScaleTierToString(sub.scaleTier) +
                " | " + scene::SimulationDomainToString(sub.domain) + "]##sub_" + sub.id;

            const ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_DefaultOpen |
                (hasNestedSubs ? ImGuiTreeNodeFlags_None : ImGuiTreeNodeFlags_Leaf);

            if (ImGui::TreeNodeEx(nodeLabel.c_str(), flags)) {
                ImGui::PushID(static_cast<int>(5000U + sIdx));
                if (ImGui::SmallButton("Zoom Into Sub-Simulation")) {
                    scene::UniverseScene subScene{};
                    std::string err;
                    if (scene::UniverseScene::LoadFromFile(sub.scenarioPath, subScene, err)) {
                        if (ctx.showFlowerOfLifeLatticeView != nullptr) {
                            *ctx.showFlowerOfLifeLatticeView = false;
                        }
                        if (ctx.onActivateScene) {
                            (void)ctx.onActivateScene(subScene, scene::SimulationScope::Scenario, std::nullopt, std::nullopt);
                        }
                    } else {
                        spdlog::warn("Could not open sub-simulation '{}': {}", sub.scenarioPath.string(), err);
                    }
                }
                ImGui::PopID();

                if (hasNestedSubs) {
                    RenderSubSimulationsForParent(ctx, sceneRef, sub.id);
                }
                ImGui::TreePop();
            }
        }
    }

    void HierarchyPanel::RenderScaleGroupedEntities(
        UIContext& ctx,
        const scene::UniverseScene& sceneRef,
        const std::vector<scene::BodyEntity>& bodies
    ) {
        std::map<scene::ScaleTier, std::vector<std::size_t>> scaleGroups;
        for (std::size_t idx = 0U; idx < bodies.size(); ++idx) {
            scaleGroups[bodies[idx].scaleTier].push_back(idx);
        }

        // Legacy descending scale order (Multiverse -> Subatomic PSU).
        static constexpr std::pair<scene::ScaleTier, const char*> kScaleCategories[] = {
            {scene::ScaleTier::Multiverse,     "[Multiverse] Scale"},
            {scene::ScaleTier::BubbleUniverse, "[Bubble Universe] Scale"},
            {scene::ScaleTier::Galaxy,         "[Galaxy] Scale"},
            {scene::ScaleTier::StarSystem,     "[Star System] Scale"},
            {scene::ScaleTier::Planet,         "[Planet] Scale"},
            {scene::ScaleTier::EntityOrDevice, "[Entity / Device] Scale"},
            {scene::ScaleTier::Atom,           "[Atom] Scale"},
            {scene::ScaleTier::SubatomicPSU,   "[Subatomic PSU] Scale"}
        };

        for (const auto& [tier, headerLabel] : kScaleCategories) {
            const auto it = scaleGroups.find(tier);
            if (it == scaleGroups.end() || it->second.empty()) {
                continue;
            }

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 0.90f, 1.0f, 1.0f));
            const bool open = ImGui::TreeNodeEx(
                headerLabel,
                ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed
            );
            ImGui::PopStyleColor();

            if (open) {
                for (const std::size_t bodyIdx : it->second) {
                    const auto& body = bodies[bodyIdx];
                    const bool isSelected =
                        (ctx.selectedBodyIndex != nullptr &&
                         ctx.selectedBodyIndex->has_value() &&
                         ctx.selectedBodyIndex->value() == bodyIdx);

                    const char* dispName = body.name.empty() ? "(Unnamed Body)" : body.name.c_str();
                    const std::string itemLabel =
                        std::string(dispName) + " [" + body.category + "]##scale_body_" + std::to_string(bodyIdx);

                    if (ImGui::Selectable(itemLabel.c_str(), isSelected, ImGuiSelectableFlags_AllowDoubleClick)) {
                        if (ctx.selectedBodyIndex != nullptr) {
                            *ctx.selectedBodyIndex = bodyIdx;
                        }
                        if (ctx.entitySelectionJustChanged != nullptr) {
                            *ctx.entitySelectionJustChanged = true;
                        }
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ctx.onFocusBodyEntity) {
                            ctx.onFocusBodyEntity(bodyIdx);
                        }
                    }
                }
                ImGui::TreePop();
            }
        }

        if (!sceneRef.subSimulations.empty()) {
            ImGui::Separator();
            ImGui::TextDisabled("Multi-Scale Sub-Simulations:");
            RenderSubSimulationsForParent(ctx, sceneRef, "");
        }
    }

    void HierarchyPanel::RenderParentChildEntities(
        UIContext& ctx,
        const scene::UniverseScene& sceneRef,
        const std::vector<scene::BodyEntity>& bodies,
        const std::string& parentId,
        bool showScaleTierTags
    ) {
        for (std::size_t idx = 0U; idx < bodies.size(); ++idx) {
            const auto& body = bodies[idx];
            if (body.parentId != parentId) {
                continue;
            }

            bool hasChildren = false;
            for (const auto& candidate : bodies) {
                if (candidate.parentId == body.id) {
                    hasChildren = true;
                    break;
                }
            }
            for (const auto& sub : sceneRef.subSimulations) {
                if (sub.parentId == body.id) {
                    hasChildren = true;
                    break;
                }
            }

            const bool isSelected =
                (ctx.selectedBodyIndex != nullptr &&
                 ctx.selectedBodyIndex->has_value() &&
                 ctx.selectedBodyIndex->value() == idx);

            ImGuiTreeNodeFlags flags =
                ImGuiTreeNodeFlags_OpenOnArrow |
                ImGuiTreeNodeFlags_SpanAvailWidth |
                ImGuiTreeNodeFlags_DefaultOpen |
                (hasChildren ? ImGuiTreeNodeFlags_None : ImGuiTreeNodeFlags_Leaf);
            if (isSelected) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }

            std::string label = body.name.empty() ? std::string("(Unnamed Body)") : body.name;
            if (showScaleTierTags) {
                label += std::string(" [") + scene::ScaleTierToString(body.scaleTier) + "]";
            } else {
                label += " [" + body.category + "]";
            }
            label += "##body_" + std::to_string(idx);

            const bool nodeOpen = ImGui::TreeNodeEx(label.c_str(), flags);
            if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !ImGui::IsItemToggledOpen()) {
                if (ctx.selectedBodyIndex != nullptr) {
                    *ctx.selectedBodyIndex = idx;
                }
                if (ctx.entitySelectionJustChanged != nullptr) {
                    *ctx.entitySelectionJustChanged = true;
                }
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ctx.onFocusBodyEntity) {
                    ctx.onFocusBodyEntity(idx);
                }
            }

            if (nodeOpen) {
                if (hasChildren) {
                    RenderParentChildEntities(ctx, sceneRef, bodies, body.id, showScaleTierTags);
                    RenderSubSimulationsForParent(ctx, sceneRef, body.id);
                }
                ImGui::TreePop();
            }
        }
    }

    void HierarchyPanel::OnUIRender(UIContext& ctx) {
        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(m_name.c_str(), &m_isOpen, m_flags)) {
            m_focused = ImGui::IsWindowFocused();
            m_hovered = ImGui::IsWindowHovered();

            // -----------------------------------------------------------------
            // Legacy Top Action Bar: "Create..." Spawner + Scale Grouping Toggle
            // -----------------------------------------------------------------
            const bool canCreateEntities =
                ctx.hasActiveScene && ctx.activeScene != nullptr &&
                ctx.activeScene->domain != scene::SimulationDomain::RadialAndScaleProfile;

            ImGui::BeginDisabled(!canCreateEntities);
            if (ImGui::Button("Create...")) {
                ImGui::OpenPopup("CreateEntityPopup");
            }
            ImGui::EndDisabled();
            RenderCreateEntityPopup(ctx);

            ImGui::SameLine();
            ImGui::Checkbox("Group by Scale", &m_groupByScaleTier);

            ImGui::Separator();

            // -----------------------------------------------------------------
            // Three-Scope Workspace Selector (Scenarios / Bubble Universes / Flower of Life)
            // -----------------------------------------------------------------
            if (ImGui::BeginTabBar("HierarchyScopeTabs")) {
                if (ImGui::BeginTabItem("Scenarios")) {
                    if (ctx.scenarios != nullptr) {
                        const auto& scenarios = ctx.scenarios->GetUniverses();
                        for (std::size_t idx = 0U; idx < scenarios.size(); ++idx) {
                            const auto& sc = scenarios[idx];
                            const bool isActiveScene =
                                (ctx.activeScope == scene::SimulationScope::Scenario) &&
                                ctx.hasActiveScene && ctx.activeScene != nullptr &&
                                (ctx.activeScene->universeId == sc.universeId);

                            ImGui::PushID(static_cast<int>(idx));
                            if (ImGui::RadioButton("##selScen", isActiveScene)) {
                                if (ctx.showFlowerOfLifeLatticeView != nullptr) {
                                    *ctx.showFlowerOfLifeLatticeView = false;
                                }
                                (void)ctx.scenarios->SetActiveUniverseIndex(idx);
                                if (ctx.onActivateScene) {
                                    (void)ctx.onActivateScene(sc, scene::SimulationScope::Scenario, std::nullopt, std::nullopt);
                                }
                            }
                            ImGui::SameLine();
                            const std::string scLabel =
                                sc.name + " [" + scene::SimulationDomainToString(sc.domain) + "]##scen_node";
                            if (ImGui::TreeNodeEx(
                                    scLabel.c_str(),
                                    isActiveScene ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None
                                )) {
                                ImGui::TextDisabled("Default: %s", scene::ModelIdToString(sc.physicsProfile.defaultModel));
                                if (isActiveScene && ctx.activeScene != nullptr) {
                                    const auto& displayBodies =
                                        (ctx.liveBodies != nullptr && ctx.liveBodies->size() == ctx.activeScene->bodies.size())
                                            ? *ctx.liveBodies
                                            : ctx.activeScene->bodies;
                                    if (!displayBodies.empty()) {
                                        if (m_groupByScaleTier) {
                                            RenderScaleGroupedEntities(ctx, *ctx.activeScene, displayBodies);
                                        } else {
                                            RenderParentChildEntities(ctx, *ctx.activeScene, displayBodies, "", false);
                                        }
                                    } else {
                                        ImGui::TextDisabled(
                                            "Analytical 1D/Scale Sweep (%u samples)",
                                            ctx.activeScene->physicsProfile.radialScale.sampleCount
                                        );
                                    }
                                } else if (!sc.bodies.empty()) {
                                    ImGui::TextDisabled("%zu bodies (select to inspect)", sc.bodies.size());
                                } else {
                                    ImGui::TextDisabled("Analytical 1D/Scale Sweep (%u samples)", sc.physicsProfile.radialScale.sampleCount);
                                }
                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                        }
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Universes")) {
                    if (ctx.universes != nullptr) {
                        const auto& univs = ctx.universes->GetUniverses();
                        for (std::size_t idx = 0U; idx < univs.size(); ++idx) {
                            const auto& u = univs[idx];
                            const bool isActiveScene =
                                (ctx.activeScope == scene::SimulationScope::BubbleUniverse) &&
                                ctx.hasActiveScene && ctx.activeScene != nullptr &&
                                (ctx.activeScene->universeId == u.universeId);

                            ImGui::PushID(static_cast<int>(1000U + idx));
                            if (ImGui::RadioButton("##selUniv", isActiveScene)) {
                                if (ctx.showFlowerOfLifeLatticeView != nullptr) {
                                    *ctx.showFlowerOfLifeLatticeView = false;
                                }
                                (void)ctx.universes->SetActiveUniverseIndex(idx);
                                if (ctx.onActivateScene) {
                                    (void)ctx.onActivateScene(u, scene::SimulationScope::BubbleUniverse, std::nullopt, std::nullopt);
                                }
                            }
                            ImGui::SameLine();
                            if (ImGui::TreeNodeEx(
                                    u.name.c_str(),
                                    isActiveScene ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None
                                )) {
                                ImGui::TextDisabled("Bubble Universe | Default: %s", scene::ModelIdToString(u.physicsProfile.defaultModel));
                                if (isActiveScene && ctx.activeScene != nullptr) {
                                    const auto& displayBodies =
                                        (ctx.liveBodies != nullptr && ctx.liveBodies->size() == ctx.activeScene->bodies.size())
                                            ? *ctx.liveBodies
                                            : ctx.activeScene->bodies;
                                    if (m_groupByScaleTier) {
                                        RenderScaleGroupedEntities(ctx, *ctx.activeScene, displayBodies);
                                    } else {
                                        RenderParentChildEntities(ctx, *ctx.activeScene, displayBodies, "", true);
                                        RenderSubSimulationsForParent(ctx, *ctx.activeScene, "");
                                    }
                                } else {
                                    ImGui::TextDisabled("%zu bodies, %zu sub-sims", u.bodies.size(), u.subSimulations.size());
                                }
                                ImGui::TreePop();
                            }
                            ImGui::PopID();
                        }
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Multiverse")) {
                    if (ctx.flowerMultiverse == nullptr || ctx.flowerMultiverse->IsEmpty()) {
                        ImGui::TextDisabled("No .multiverse.umgebung manifest loaded.");
                    } else {
                        ImGui::TextWrapped("%s", ctx.flowerMultiverse->GetName().c_str());
                        const auto& vs = ctx.flowerMultiverse->GetVoidState();
                        ImGui::TextDisabled(
                            "The Void: cond=%.2f | Love=%.0f Hz | Phryll=%.0f",
                            vs.bridgeConductivity, vs.loveBindingFrequency, vs.sourcePhryllPotential
                        );
                        if (ctx.showFlowerOfLifeLatticeView != nullptr) {
                            ImGui::Checkbox("Show Lattice in Viewport", ctx.showFlowerOfLifeLatticeView);
                        }
                        ImGui::Separator();

                        const auto& nodes = ctx.flowerMultiverse->GetNodes();
                        for (std::size_t idx = 0U; idx < nodes.size(); ++idx) {
                            const auto& node = nodes[idx];
                            if (!node.loadedUniverse.has_value()) {
                                continue;
                            }
                            const bool isSelectedNode =
                                (idx == ctx.flowerMultiverse->GetActiveNodeIndex() &&
                                 ctx.activeScope == scene::SimulationScope::FlowerOfLifeMultiverse);
                            ImGui::PushID(static_cast<int>(2000U + idx));
                            if (ImGui::SmallButton(isSelectedNode ? "Active" : "Enter")) {
                                (void)ctx.flowerMultiverse->SetActiveNodeIndex(idx);
                                if (ctx.showFlowerOfLifeLatticeView != nullptr) {
                                    *ctx.showFlowerOfLifeLatticeView = false;
                                }
                                if (ctx.onActivateScene) {
                                    (void)ctx.onActivateScene(
                                        node.loadedUniverse.value(),
                                        scene::SimulationScope::FlowerOfLifeMultiverse,
                                        std::nullopt,
                                        std::nullopt
                                    );
                                }
                            }
                            ImGui::SameLine();
                            ImGui::Text("[Ring %u] %s", node.ringIndex, node.label.c_str());
                            ImGui::PopID();
                        }
                    }
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            // Clicking empty whitespace in Hierarchy clears entity selection (legacy UX).
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                ImGui::IsWindowHovered() &&
                !ImGui::IsAnyItemHovered()) {
                if (ctx.selectedBodyIndex != nullptr) {
                    ctx.selectedBodyIndex->reset();
                }
            }
        }
        ImGui::End();
    }

} // namespace umgebung::ui
