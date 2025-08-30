#pragma once

#include "umgebung/util/Config.hpp"
#include <string>
#include <map>
#include <glm/glm.hpp>

namespace Umgebung {
    namespace renderer {
        class Camera {
        public:
            Camera(util::ConfigManager& configManager, float width, float height);
            void setCurrentZoomLevel(const std::string& levelName);

            glm::mat4 getViewMatrix() const;
            glm::mat4 getProjectionMatrix() const;

        private:
            util::ConfigManager& m_configManager;
            std::string m_currentLevelName;
            float m_nearPlane = 0.1f;
            float m_farPlane = 100.0f;
            float m_width;
            float m_height;

            glm::vec3 m_position = glm::vec3(0.0f, 0.0f, 3.0f);
            glm::vec3 m_front = glm::vec3(0.0f, 0.0f, -1.0f);
            glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f);
        };
    }
}