/**
 * @file HierarchyPanel.hpp
 * @brief Contains the HierarchyPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"

namespace Umgebung::scene { class Scene; }

namespace Umgebung::ui::imgui {

    /**
     * @brief A class for the hierarchy panel.
     */
    class HierarchyPanel : public Panel {
    public:
        /**
         * @brief Construct a new Hierarchy Panel object.
         * 
         * @param scene The scene to display.
         */
        explicit HierarchyPanel(scene::Scene* scene);

        /**
         * @brief Renders the hierarchy panel.
         */
        void onUIRender() override;

    private:
        scene::Scene* scene_ = nullptr; ///< The scene to display.
    };

}