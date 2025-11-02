#pragma once
#include <vector>
#include "Planet.hpp"
#include "Star.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
			/**
 * @file SolarSystem.hpp
 * @brief Contains the SolarSystem class.
 */
#pragma once
#include <vector>
#include "Planet.hpp"
#include "Star.hpp"

namespace Umgebung {
	namespace ecs {
		namespace entities {
            /**
             * @brief A class representing a solar system.
             * 
             * This class contains a vector of planets and stars.
             */
			class SolarSystem {
			public:
                /**
                 * @brief Construct a new Solar System object.
                 */
				SolarSystem();

                /**
                 * @brief Destroy the Solar System object.
                 */
				~SolarSystem();

                /**
                 * @brief Get the Planets object.
                 * 
                 * @return const std::vector<Planet>& 
                 */
				const std::vector<Planet>& getPlanets() const { return m_Planets; }

                /**
                 * @brief Get the Stars object.
                 * 
                 * @return const std::vector<Star>& 
                 */
				const std::vector<Star>& getStars() const { return m_Stars; }

                /**
                 * @brief Set the Planets object.
                 * 
                 * @param planets 
                 */
				void setPlanets(const std::vector<Planet>& planets) { m_Planets = planets; }

                /**
                 * @brief Set the Stars object.
                 * 
                 * @param stars 
                 */
				void setStars(const std::vector<Star>& stars) { m_Stars = stars; }

			private:
				std::vector<Planet> m_Planets; ///< The planets in the solar system.
				std::vector<Star> m_Stars;     ///< The stars in the solar system.
			};
		}
	}
}
		}
	}
}