#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::scene { class Scene; }

namespace Umgebung::ui::imgui {

    class HierarchyPanel : public Panel {
    public:
        explicit HierarchyPanel(scene::Scene* scene);

        void onUIRender() override;

    private:
        scene::Scene* scene_ = nullptr;
    };

}