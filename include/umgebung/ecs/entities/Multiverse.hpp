/**
 * @file Multiverse.hpp
 * @brief Contains the Multiverse class.
 */
#pragma once
#include <vector>
#include "Universe.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
            /**
             * @brief A class representing a multiverse.
             * 
             * This class contains a vector of universes.
             */
			class Multiverse {
			public:
                /**
                 * @brief Construct a new Multiverse object.
                 */
				Multiverse();

                /**
                 * @brief Destroy the Multiverse object.
                 */
				~Multiverse();

                /**
                 * @brief Get the Universes object.
                 * 
                 * @return const std::vector<Universe>& 
                 */
				const std::vector<Universe>& getUniverses() const { return m_Universes; }

                /**
                 * @brief Set the Universes object.
                 * 
                 * @param universes 
                 */
				void setUniverses(const std::vector<Universe>& universes) { m_Universes = universes; }
			private:
				std::vector<Universe> m_Universes; ///< The universes in the multiverse.
			};
		}
	}
}