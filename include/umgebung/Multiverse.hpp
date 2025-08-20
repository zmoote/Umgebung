#pragma once
#include <vector>
#include "umgebung/Universe.hpp"

namespace Umgebung {
	class Multiverse {
	public:
		Multiverse();
		~Multiverse();

		const std::vector<Universe>& getUniverses() const { return m_Universes; }

		void setUniverses(const std::vector<Universe>& universes) { m_Universes = universes; }
	private:
		std::vector<Umgebung::Universe> m_Universes;
	};
}