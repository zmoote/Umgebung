/**
 * @file Universe.hpp
 * @brief Contains the Universe class.
 */
#pragma once
#include <vector>
#include "Galaxy.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
            /**
             * @brief A class representing a universe.
             * 
             * This class contains a vector of galaxies.
             */
				class Universe {
				public:
                    /**
                     * @brief Construct a new Universe object.
                     */
					Universe();

                    /**
                     * @brief Destroy the Universe object.
                     */
					~Universe();

                    /**
                     * @brief Get the Galaxies object.
                     * 
                     * @return const std::vector<Galaxy>& 
                     */
					const std::vector<Galaxy>& getGalaxies() const { return m_Galaxies; }

                    /**
                     * @brief Set the Galaxies object.
                     * 
                     * @param galaxies 
                     */
					void setGalaxies(const std::vector<Galaxy>& galaxies) { m_Galaxies = galaxies; }

				private:
					std::vector<Galaxy> m_Galaxies; ///< The galaxies in the universe.
				};
		}
	}
}