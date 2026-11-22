/**
 * @file HierarchyPanel.hpp
 * @brief Docked left-pane `"Hierarchy"` panel combining legacy Umgebung's
 *        `Create...` entity spawner and scale-grouped selectable entity tree
 *        with current Umgebung's three-scope workspace (`Scenarios`,
 *        `Bubble Universes`, `Flower of Life`).
 *
 * @details
 * Docked at `20%` width on the left side of `MainDockSpace` (matching every
 * branch of `Umgebung-legacy`). Clicking an entity selects it for editing in
 * the right-docked `PropertiesPanel`; double-clicking an entity centers/focuses
 * the `ViewportPanel` on that body; clicking empty whitespace clears selection
 * so `PropertiesPanel` displays the active scene's Simulation & A/B controls.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/HierarchyPanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/scene/UniverseScene.hpp"
#include "umgebung/ui/Panel.hpp"

#include <string>
#include <vector>

namespace umgebung::ui {

    /**
     * @class HierarchyPanel
     * @brief Left-docked scene, scope, and multi-scale entity hierarchy panel.
     */
    class HierarchyPanel final : public Panel {
    public:
        /** @brief Constructs the `"Hierarchy"` panel (open by default on the left dock). */
        HierarchyPanel();

        /**
         * @brief Renders the `"Hierarchy"` panel with `Create...` popup,
         *        three-scope tabs, scale-grouped / parent-child entity tree,
         *        and multi-scale sub-simulation links.
         *
         * @param ctx Shared UI context providing workspace scenes and callbacks.
         */
        void OnUIRender(UIContext& ctx) override;

    private:
        /**
         * @brief Renders the legacy-inspired `"Create..."` popup menu spawning
         *        typed bodies/devices into the active scene across scale tiers.
         */
        void RenderCreateEntityPopup(UIContext& ctx);

        /**
         * @brief Renders active scene bodies grouped by `scene::ScaleTier`
         *        in descending cosmological-to-subatomic order (legacy style).
         */
        void RenderScaleGroupedEntities(
            UIContext& ctx,
            const scene::UniverseScene& sceneRef,
            const std::vector<scene::BodyEntity>& bodies
        );

        /**
         * @brief Recursively renders parent-child entity nodes and nested
         *        `ScaleTierSubSimulation` entries inside a scene or Bubble Universe.
         */
        void RenderParentChildEntities(
            UIContext& ctx,
            const scene::UniverseScene& sceneRef,
            const std::vector<scene::BodyEntity>& bodies,
            const std::string& parentId,
            bool showScaleTierTags
        );

        /**
         * @brief Renders domain-appropriate `ScaleTierSubSimulation` zoom links
         *        attached to `parentId` inside a Bubble Universe.
         */
        void RenderSubSimulationsForParent(
            UIContext& ctx,
            const scene::UniverseScene& sceneRef,
            const std::string& parentId
        );

        /** @brief Toggle between legacy Scale-Grouped list (`true`) and Parent-Child Tree (`false`). */
        bool m_groupByScaleTier{false};
    };

} // namespace umgebung::ui
