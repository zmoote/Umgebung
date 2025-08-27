#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class HierarchyPanel
             * @brief A concrete panel that lists and manages Entities within a Hierarchy.
             */
            class HierarchyPanel : public Panel {
            public:
                HierarchyPanel();
                void render() override;
            };
        }
    }
}