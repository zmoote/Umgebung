#pragma once
#include <vector>
#include "umgebung/Galaxy.hpp"

namespace Umgebung {
	class Universe {
	public:
		Universe();
		~Universe();

		const std::vector<Galaxy>& getGalaxies() const { return m_Galaxies; }

		void setGalaxies(const std::vector<Galaxy>& galaxies) { m_Galaxies = galaxies; }

	private:
		std::vector<Umgebung::Galaxy> m_Galaxies;
	};
}