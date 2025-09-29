#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            class StatisticsPanel : public Panel {
            public:
                StatisticsPanel();
                void onUIRender() override;
            };
        }
    }
}