#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::ui::imgui {

    class AboutPanel : public Panel {
    public:
        AboutPanel();

        void onUIRender() override;
    };

}