#pragma once
#include <vector>
#include "SolarSystem.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
			class Galaxy {
			public:
				Galaxy();
				~Galaxy();

				const std::vector<SolarSystem>& getSolarSystems() const { return m_SolarSystems; }

				void setSolarSystems(const std::vector<SolarSystem>& solarsystems) { m_SolarSystems = solarsystems; }

			private:
				std::vector<SolarSystem> m_SolarSystems;
			};
		}
	}
}