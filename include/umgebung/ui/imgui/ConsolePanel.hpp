#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            class ConsolePanel : public Panel {
            public:
                ConsolePanel();
                void onUIRender() override;
            };
        }
    }
}