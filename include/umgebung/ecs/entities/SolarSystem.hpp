#pragma once
#include <vector>
#include "Planet.hpp"
#include "Star.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
			class SolarSystem {
			public:
				SolarSystem();
				~SolarSystem();

				const std::vector<Planet>& getPlanets() const { return m_Planets; }
				const std::vector<Star>& getStars() const { return m_Stars; }

				void setPlanets(const std::vector<Planet>& planets) { m_Planets = planets; }
				void setStars(const std::vector<Star>& stars) { m_Stars = stars; }

			private:
				std::vector<Planet> m_Planets;
				std::vector<Star> m_Stars;
			};
		}
	}
}