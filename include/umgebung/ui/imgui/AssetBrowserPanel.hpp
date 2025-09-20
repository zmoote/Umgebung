#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @class AssetBrowserPanel
             * @brief A concrete panel that lists the contents of the assets/ directory
             */
            class AssetBrowserPanel : public Panel {
            public:
                AssetBrowserPanel();
                void onUIRender() override;
            };
        }
    }
}