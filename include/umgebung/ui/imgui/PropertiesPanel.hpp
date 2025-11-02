#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::scene { class Scene; }

namespace Umgebung::ui::imgui {

    /**
 * @file PropertiesPanel.hpp
 * @brief Contains the PropertiesPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::scene { class Scene; }

namespace Umgebung::ui::imgui {

    /**
     * @brief A class for the properties panel.
     */
    class PropertiesPanel : public Panel {
    public:
        /**
         * @brief Construct a new Properties Panel object.
         * 
         * @param scene The scene to get the properties from.
         */
        explicit PropertiesPanel(scene::Scene* scene);

        /**
         * @brief Renders the properties panel.
         */
        void onUIRender() override;

    private:
        scene::Scene* scene_ = nullptr; ///< The scene to get the properties from.
    };

}

}