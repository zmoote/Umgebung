#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

// Forward-declare Scene
namespace Umgebung::scene { class Scene; }

namespace Umgebung::ui::imgui {

    class PropertiesPanel : public Panel {
    public:
        // The constructor now just takes the scene and passes a name to the base class
        explicit PropertiesPanel(scene::Scene* scene);

        // The function name is changed from render to onUIRender
        void onUIRender() override;

    private:
        scene::Scene* scene_ = nullptr;
    };

} // namespace Umgebung::ui::imgui