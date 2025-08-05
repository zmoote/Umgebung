#pragma once
#include "Star.hpp"
#include "Planet.hpp"
#include <vector>

namespace Umgebung {
    class SolarSystem {
    protected:
        Star* star;
        std::vector<Planet*> planets;
    public:
        SolarSystem() : star(nullptr) {}
        virtual ~SolarSystem() {
            delete star;
            for (auto* p : planets) delete p;
        }

        void setStar(Star* s) { star = s; }
        void addPlanet(Planet* p) { planets.push_back(p); }
        size_t getPlanetCount() const { return planets.size(); }
    };
}