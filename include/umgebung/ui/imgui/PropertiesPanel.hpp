#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <functional>
#include <string>
#include <vector>
#include <filesystem>
#include <memory>

namespace Umgebung {
    namespace scene {
        class Scene;
    }
    namespace ui {
        namespace imgui {

            class PropertiesPanel : public Panel {
            public:
                using OpenFilePickerFn = std::function<void(const std::string&, const std::string&, std::function<void(const std::filesystem::path&)>, const std::vector<std::string>&)>;
                PropertiesPanel(scene::Scene* scene, OpenFilePickerFn openFilePicker);

                void onUIRender() override;

            private:
                scene::Scene* scene_;
                OpenFilePickerFn openFilePicker_;
            };

        }
    }
}