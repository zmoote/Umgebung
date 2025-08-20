#pragma once
#include <vector>
#include "umgebung/SolarSystem.hpp"

namespace Umgebung {
	class Galaxy {
	public:
		Galaxy();
		~Galaxy();

		const std::vector<SolarSystem>& getSolarSystems() const { return m_SolarSystems; }

		void setSolarSystems(const std::vector<SolarSystem>& solarsystems) { m_SolarSystems = solarsystems; }

	private:
		std::vector<Umgebung::SolarSystem> m_SolarSystems;
	};
}