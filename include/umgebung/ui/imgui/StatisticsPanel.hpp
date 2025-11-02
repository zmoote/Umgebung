#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
 * @file StatisticsPanel.hpp
 * @brief Contains the StatisticsPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @brief A class for the statistics panel.
             */
            class StatisticsPanel : public Panel {
            public:
                /**
                 * @brief Construct a new Statistics Panel object.
                 */
                StatisticsPanel();

                /**
                 * @brief Renders the statistics panel.
                 */
                void onUIRender() override;
            };
        }
    }
}
        }
    }
}