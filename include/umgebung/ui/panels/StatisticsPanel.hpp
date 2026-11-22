/**
 * @file StatisticsPanel.hpp
 * @brief Floating non-docking "Statistics" utility panel opened from `Tools -> Statistics`.
 *
 * @details
 * Preserves the exact non-docking, auto-sized floating telemetry window from
 * `Umgebung-legacy` (`Tools -> Statistics`) while integrating current Umgebung's
 * Hardware & Adaptation readout (Hardware Tier, CPU, Host RAM, Vulkan GPU &
 * CUDA-UUID match, CUDA streams, tracked VRAM budget, and live simulation
 * stepping telemetry).
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/StatisticsPanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/Panel.hpp"

namespace umgebung::ui {

    /**
     * @class StatisticsPanel
     * @brief Floating telemetry window displaying live FPS, frame timing,
     *        hardware discovery/adaptation state, CUDA VRAM usage, and solver pacing.
     */
    class StatisticsPanel final : public Panel {
    public:
        /** @brief Constructs a `StatisticsPanel` (closed by default; opened via `Tools -> Statistics`). */
        StatisticsPanel();

        /**
         * @brief Renders the floating `"Statistics"` window when open.
         *
         * @param ctx Shared UI context supplying hardware, CUDA, and frame/solver stats.
         */
        void OnUIRender(UIContext& ctx) override;
    };

} // namespace umgebung::ui
