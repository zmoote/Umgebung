#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            class AssetBrowserPanel : public Panel {
            public:
                AssetBrowserPanel();
                void onUIRender() override;
            };
        }
    }
}