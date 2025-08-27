#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class StatisticsPanel
             * @brief A concrete panel that displays performance statistics.
             */
            class StatisticsPanel : public Panel {
            public:
                StatisticsPanel();
                void render() override;
            };
        }
    }
}