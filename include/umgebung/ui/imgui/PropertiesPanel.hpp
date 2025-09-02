#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class PropertiesPanel
             * @brief A concrete panel that lists and manages components of Entities
             */
            class PropertiesPanel : public Panel {
            public:
                PropertiesPanel();
                void render() override;
            };
        }
    }
}