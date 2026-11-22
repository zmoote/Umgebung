/**
 * @file PropertiesPanel.hpp
 * @brief Docked right-pane `"Properties"` inspector panel adapting legacy
 *        Umgebung's `CollapsingHeader` entity inspector together with current
 *        Umgebung's domain-filtered Model Registry, live Thinker/Device controls,
 *        playback pacing, and Simultaneous A/B Comparison telemetry.
 *
 * @details
 * Docked at `25%` width on the right side of `MainDockSpace` (matching
 * `Umgebung-legacy`'s `origin/dev` layout).
 *   - When an entity is selected in `HierarchyPanel`, `PropertiesPanel` offers
 *     two clean tabs (`Entity` and `Simulation & A/B`), auto-focusing `Entity`
 *     upon fresh selection so users can immediately rename the body, edit
 *     transform/kinematics/mass/radius/scale tier/vibrational density (with
 *     legacy's exact 1D–13D resonance plane readouts), inspect live per-body
 *     A/B divergence, or delete the entity.
 *   - When no entity is selected (including on initial launch), `PropertiesPanel`
 *     directly displays the active scene's `Simulation & A/B` collapsible
 *     inspector sections (Primary Model A + Provenance badge, Overlay Model B,
 *     domain/Thinker pilot controls, playback & wall-clock pacing, live telemetry,
 *     and A/B divergence).
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/PropertiesPanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/Panel.hpp"

#include <cstddef>

namespace umgebung::ui {

    /**
     * @class PropertiesPanel
     * @brief Right-docked inspector panel for both selected entities and active
     *        simulation / physics / A/B comparison configuration.
     */
    class PropertiesPanel final : public Panel {
    public:
        /** @brief Constructs the `"Properties"` panel (open by default on the right dock). */
        PropertiesPanel();

        /**
         * @brief Renders the `"Properties"` window and its collapsible sections.
         *
         * @param ctx Shared UI context supplying active scene, selection, and telemetry.
         */
        void OnUIRender(UIContext& ctx) override;

    private:
        /**
         * @brief Renders the legacy-inspired `CollapsingHeader` entity inspector
         *        for `ctx.activeScene->bodies[bodyIndex]`.
         */
        void RenderEntityInspector(UIContext& ctx, std::size_t bodyIndex);

        /**
         * @brief Renders the Simulation, Domain-Filtered Model Registry,
         *        Thinker/Device Pilot Controls, Playback Pacing, Live Telemetry,
         *        and Simultaneous A/B Divergence collapsible sections.
         */
        void RenderSimulationInspector(UIContext& ctx);
    };

} // namespace umgebung::ui
