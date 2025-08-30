#pragma once
#include "umgebung/util/Config.hpp"

namespace Umgebung {
	namespace renderer {
		class Camera {
		public:
			Camera(util::ConfigManager& configManager);
			void setCurrentZoomLevel(const std::string& levelName);
		private:
			util::ConfigManager& m_configManager;
			std::string m_currentLevelName;
			float m_nearPlane;
			float m_farPlane;
			std::string m_units;
		};
	}
}