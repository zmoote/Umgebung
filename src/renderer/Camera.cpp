#include "umgebung/renderer/Camera.hpp"
#include "umgebung/util/LogMacros.hpp"

namespace Umgebung {
	namespace renderer {

		Camera::Camera(util::ConfigManager& configManager) 
			: m_configManager(configManager)
		{

		}

		void Camera::setCurrentZoomLevel(const std::string& levelName) 
		{
            // Get the map of all available levels
            const auto& allLevels = m_configManager.getCameraLevels();

            // Try to find the level the user asked for
            auto it = allLevels.find(levelName);

            // Check if the find operation was successful
            if (it != allLevels.end())
            {
                // --- We found it! ---

                // Get the actual CameraLevel struct from the iterator
                const util::CameraLevel& foundLevel = it->second;

                // Apply the settings to our camera's member variables
                this->m_nearPlane = foundLevel.nearPlane;
                this->m_farPlane = foundLevel.farPlane;

                // And, as you correctly pointed out, store the name of the new level
                this->m_currentLevelName = levelName;
            }
            else
            {
                // --- We didn't find it! ---
                // It's good practice to log an error here so we know if we
                // typed a level name wrong.
                UMGEBUNG_LOG_ERROR("Couldn't find a CameraLevel with name {}", levelName);
            }
		}
	}
}