#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::ui::imgui {

    class AboutPanel : public Panel {
    public:
        // This constructor takes no arguments
        AboutPanel();

        void onUIRender() override;
    };

} // namespace Umgebung::ui::imgui