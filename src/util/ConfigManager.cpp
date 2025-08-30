#include "umgebung/util/Config.hpp"

namespace Umgebung {
	namespace util {
		void ConfigManager::loadConfig(const std::string& filepath)
		{
			std::ifstream file;
			nlohmann::json json;

			file.open(filepath);

			file >> json;

			for (auto& [key, value] : json["cameraLevels"].items()) 
			{
				CameraLevel level;
				level.farPlane = value["farPlane"];
				level.nearPlane = value["nearPlane"];
				level.units = value["units"];

				cameraLevels.emplace(key, level);
			} 
		}
	}
}