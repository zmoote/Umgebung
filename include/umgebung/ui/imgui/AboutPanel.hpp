#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class AboutPanel
             * @brief A concrete panel that displays information about Umgebung.
             */
            class AboutPanel : public Panel {
            public:
                AboutPanel();
                void render() override;
            };
        }
    }
}