#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class PropertiesPanel
             * @brief A concrete panel that displays Logs from my Logger class, specifically the 
             * LogMacros
             */
            class ConsolePanel : public Panel {
            public:
                ConsolePanel();
                void render() override;
            };
        }
    }
}