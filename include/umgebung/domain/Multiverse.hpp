#pragma once
#include <vector>
#include "Universe.hpp"

namespace Umgebung {
	namespace domain {
		class Multiverse {
		public:
			Multiverse();
			~Multiverse();

			const std::vector<Universe>& getUniverses() const { return m_Universes; }

			void setUniverses(const std::vector<Universe>& universes) { m_Universes = universes; }
		private:
			std::vector<Universe> m_Universes;
		};
	}
}