/**
 * @file AboutPanel.hpp
 * @brief Floating non-docking "About Umgebung" dialog panel opened from `Help -> About Umgebung`.
 *
 * @details
 * Preserves the exact non-docking, non-collapsible, non-resizable floating utility
 * style of `Umgebung-legacy`'s `AboutPanel` while displaying current Umgebung
 * engine version, build configuration, and provenance information.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/panels/AboutPanel.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/ui/Panel.hpp"

namespace umgebung::ui {

    /**
     * @class AboutPanel
     * @brief Floating utility window presenting Umgebung credits, origin, and
     *        active graphics/compute runtime summary.
     */
    class AboutPanel final : public Panel {
    public:
        /** @brief Constructs an `AboutPanel` (closed by default until triggered from `Help`). */
        AboutPanel();

        /**
         * @brief Renders the floating "About" window if open.
         *
         * @param ctx Shared UI context supplying engine version and GPU summary.
         */
        void OnUIRender(UIContext& ctx) override;
    };

} // namespace umgebung::ui
