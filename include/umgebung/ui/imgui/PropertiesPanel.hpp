/**
 * @file PropertiesPanel.hpp
 * @brief Contains the PropertiesPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <memory>

namespace Umgebung {
    namespace scene {
        class Scene;
    }
    namespace ui {
        namespace imgui {
            class FilePickerPanel;

            class PropertiesPanel : public Panel {
            public:
                PropertiesPanel(scene::Scene* scene);

                void onUIRender() override;

            private:
                scene::Scene* scene_;
                std::unique_ptr<FilePickerPanel> filePicker_;
            };

        }
    }
}