#pragma once
#include <vector>
#include "Galaxy.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
				class Universe {
				public:
					Universe();
					~Universe();

					const std::vector<Galaxy>& getGalaxies() const { return m_Galaxies; }

					void setGalaxies(const std::vector<Galaxy>& galaxies) { m_Galaxies = galaxies; }

				private:
					std::vector<Galaxy> m_Galaxies;
				};
		}
	}
}