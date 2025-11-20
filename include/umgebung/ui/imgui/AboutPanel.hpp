/**
 * @file AboutPanel.hpp
 * @brief Contains the AboutPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::ui::imgui {

    /**
     * @brief A class for the about panel.
     */
    class AboutPanel : public Panel {
    public:
        /**
         * @brief Construct a new About Panel object.
         */
        AboutPanel();

        /**
         * @brief Renders the about panel.
         */
        void onUIRender() override;
    };

}