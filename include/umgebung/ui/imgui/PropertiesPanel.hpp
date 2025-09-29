#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::scene { class Scene; }

namespace Umgebung::ui::imgui {

    class PropertiesPanel : public Panel {
    public:
        explicit PropertiesPanel(scene::Scene* scene);

        void onUIRender() override;

    private:
        scene::Scene* scene_ = nullptr;
    };

}