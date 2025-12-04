#pragma once

#include "umgebung/renderer/Camera.hpp"
#include "umgebung/ecs/components/ScaleComponent.hpp"
#include <nlohmann/json.hpp>
#include <map>
#include <string>

namespace Umgebung::ecs::systems {

    struct CameraLevelConfig {
        float nearPlane;
        float farPlane;
        std::string units;
    };

    class ObserverSystem {
    public:
        ObserverSystem();
        ~ObserverSystem();

        void init();
        void onUpdate(renderer::Camera& camera);

        components::ScaleType getCurrentScale() const { return currentScale_; }

    private:
        void loadConfig();
        void updateCameraSettings(renderer::Camera& camera);

        std::map<components::ScaleType, CameraLevelConfig> config_;
        components::ScaleType currentScale_ = components::ScaleType::Human;
        bool firstUpdate_ = true;
    };

}
