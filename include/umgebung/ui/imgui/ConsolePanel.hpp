/**
 * @file ConsolePanel.hpp
 * @brief Contains the ConsolePanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @brief A class for the console panel.
             */
            class ConsolePanel : public Panel {
            public:
                /**
                 * @brief Construct a new Console Panel object.
                 */
                ConsolePanel();

                /**
                 * @brief Renders the console panel.
                 */
                void onUIRender() override;
            };
        }
    }
}