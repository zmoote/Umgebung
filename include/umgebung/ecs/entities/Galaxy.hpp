/**
 * @file Galaxy.hpp
 * @brief Contains the Galaxy class.
 */
#pragma once
#include <vector>
#include "SolarSystem.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
            /**
             * @brief A class representing a galaxy.
             * 
             * This class contains a vector of solar systems.
             */
			class Galaxy {
			public:
                /**
                 * @brief Construct a new Galaxy object.
                 */
				Galaxy();

                /**
                 * @brief Destroy the Galaxy object.
                 */
				~Galaxy();

                /**
                 * @brief Get the Solar Systems object.
                 * 
                 * @return const std::vector<SolarSystem>& 
                 */
				const std::vector<SolarSystem>& getSolarSystems() const { return m_SolarSystems; }

                /**
                 * @brief Set the Solar Systems object.
                 * 
                 * @param solarsystems 
                 */
				void setSolarSystems(const std::vector<SolarSystem>& solarsystems) { m_SolarSystems = solarsystems; }

			private:
				std::vector<SolarSystem> m_SolarSystems; ///< The solar systems in the galaxy.
			};
		}
	}
}