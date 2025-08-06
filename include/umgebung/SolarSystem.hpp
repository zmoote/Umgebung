#pragma once
#include "Star.hpp"
#include "Planet.hpp"
#include <vector>

namespace Umgebung {
    class SolarSystem {
    protected:
        std::vector<Star*> stars;
        std::vector<Planet*> planets;
    public:
        SolarSystem();
        virtual ~SolarSystem() {
            for (auto* s : stars) delete s;
            for (auto* p : planets) delete p;
        }

        void addStar(Star* s) { stars.push_back(s); }
        void addPlanet(Planet* p) { planets.push_back(p); }
        size_t getPlanetCount() const { return planets.size(); }
        size_t getStarCount() const { return stars.size(); }
    };
}