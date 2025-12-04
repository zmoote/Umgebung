/**
 * @file HierarchyPanel.hpp
 * @brief Contains the HierarchyPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <functional>
#include <entt/fwd.hpp>

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
         * @param onEntityFocusCallback Optional callback for when an entity is double-clicked.
         */
        explicit HierarchyPanel(scene::Scene* scene, std::function<void(entt::entity)> onEntityFocusCallback = nullptr);

        /**
         * @brief Renders the hierarchy panel.
         */
        void onUIRender() override;

    private:
        scene::Scene* scene_ = nullptr; ///< The scene to display.
        std::function<void(entt::entity)> onEntityFocusCallback_;
    };

}