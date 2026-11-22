/**
 * @file ViewportPanel.hpp
 * @brief Center-docked `"Viewport"` panel adapting legacy Umgebung's zero-padding
 *        central viewport canvas (with subtle playback-state indicator and focus
 *        tracking) to current Umgebung's four domain-adaptive visualizers.
 *
 * @details
 * Docked into the central node (`dock_main_id`) of `MainDockSpace` across all
 * `Umgebung-legacy` branches. Supports:
 *   1. `FlowerOfLifeMultiverse` interactive geometric Bubble Universe lattice
 *      (click any populated node to enter its Bubble Universe),
 *   2. `NBodyDynamics` 2D orbital trajectory canvas with live mouse-wheel zoom,
 *      drag panning, entity double-click focusing, solid Model A trails, dashed
 *      Model B ghost trails, and per-body divergence vectors,
 *   3. `FieldGrid2D` GPU field heatmap (`Model A`, `Model B`, or `|A - B|`),
 *      stationary device geometry overlays (HEEMFG ring, Vogel Phryll crystal,
 *      GE C204 22m×6m envelope + 170° X-ray cone + Lense-Thirring streamlines +
 *      VGL scan pulse), and live center-probe waveform strip,
 *   4. `RadialAndScaleProfile` analytical log-log / linear curve & residual plot
 *      with physical landmark vertical markers.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/ViewportPanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/Panel.hpp"

#include <imgui.h>

namespace umgebung::ui {

    /**
     * @class ViewportPanel
     * @brief Central docked simulation visualization viewport panel.
     */
    class ViewportPanel final : public Panel {
    public:
        /** @brief Constructs the `"Viewport"` panel (open by default in center dock). */
        ViewportPanel();

        /**
         * @brief Renders the zero-padded `"Viewport"` window, compact top toolbar,
         *        playback state border indicator, and active domain canvas.
         *
         * @param ctx Shared UI context supplying domain snapshots and camera state.
         */
        void OnUIRender(UIContext& ctx) override;

        /** @brief Returns the most recent available content size `(width, height)` in pixels. */
        [[nodiscard]] ImVec2 GetViewportSize() const noexcept { return m_viewportSize; }

    private:
        /** @brief Renders the compact Play/Pause/Stop/Step/Reset & domain bar at top of Viewport. */
        void RenderViewportToolbar(UIContext& ctx);

        /** @brief Renders Elena Danaan's interactive Flower of Life Bubble Universe lattice. */
        void RenderFlowerOfLifeCanvas(UIContext& ctx);

        /** @brief Renders the interactive 2D `NBodyDynamics` orbital trajectory canvas. */
        void RenderOrbitalCanvas(UIContext& ctx);

        /** @brief Renders the 2D `FieldGrid2D` GPU heatmap, device overlays, and probe waveform. */
        void RenderFieldGrid2DCanvas(UIContext& ctx);

        /** @brief Renders the 1D `RadialAndScaleProfile` analytical curve & residual plotter. */
        void RenderRadialScaleCanvas(UIContext& ctx);

        /** @brief Last measured content-region dimensions inside `"Viewport"`. */
        ImVec2 m_viewportSize{0.0f, 0.0f};
    };

} // namespace umgebung::ui
