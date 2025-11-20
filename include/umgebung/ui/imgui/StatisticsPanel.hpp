/**
 * @file StatisticsPanel.hpp
 * @brief Contains the StatisticsPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/ecs/systems/DebugRenderSystem.hpp"

namespace Umgebung {
    namespace ui {
        namespace imgui {
            /**
             * @brief A class for the statistics panel.
             */
            class StatisticsPanel : public Panel {
            public:
                /**
                 * @brief Construct a new Statistics Panel object.
                 */
                StatisticsPanel(ecs::systems::DebugRenderSystem* debugRenderSystem);

                /**
                 * @brief Renders the statistics panel.
                 */
                void onUIRender() override;
            
            private:
                ecs::systems::DebugRenderSystem* debugRenderSystem_;
            };
        }
    }
}