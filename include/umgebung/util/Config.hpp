#pragma once

#include <string>
#include <map>
#include <fstream>
#include <nlohmann/json.hpp>

namespace Umgebung {
    namespace util {

        struct CameraLevel {
            float nearPlane;
            float farPlane;
            std::string units;
        };

        class ConfigManager {
        public:
            const std::map<std::string, CameraLevel>& getCameraLevels() const {
                return cameraLevels;
            }

            void loadConfig(const std::string& filePath);

        private:
            std::map<std::string, CameraLevel> cameraLevels;
        };

    } // namespace util
} // namespace umgebung