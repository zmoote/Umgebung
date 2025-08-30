#include "umgebung/renderer/Camera.hpp"
#include "umgebung/util/LogMacros.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung {
    namespace renderer {

        Camera::Camera(util::ConfigManager& configManager, float width, float height)
            : m_configManager(configManager), m_width(width), m_height(height)
        {
        }

        void Camera::setCurrentZoomLevel(const std::string& levelName)
        {
            const auto& allLevels = m_configManager.getCameraLevels();
            auto it = allLevels.find(levelName);
            if (it != allLevels.end())
            {
                const util::CameraLevel& foundLevel = it->second;
                this->m_nearPlane = foundLevel.nearPlane;
                this->m_farPlane = foundLevel.farPlane;
                this->m_currentLevelName = levelName;
            }
            else
            {
                UMGEBUNG_LOG_ERROR("Couldn't find camera level '{}'", levelName);
            }
        }

        glm::mat4 Camera::getViewMatrix() const {
            return glm::lookAt(m_position, m_position + m_front, m_up);
        }

        glm::mat4 Camera::getProjectionMatrix() const {
            return glm::perspective(glm::radians(45.0f), m_width / m_height, m_nearPlane, m_farPlane);
        }

    } // namespace renderer
} // namespace Umgebung