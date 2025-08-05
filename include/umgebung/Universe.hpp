#pragma once
#include "Galaxy.hpp"
#include <vector>

namespace Umgebung {
    class Universe {
    protected:
        std::vector<Galaxy*> galaxies;
        int densityCount; // Number of densities levels
    public:
        Universe(int dc) : densityCount(dc) {}
        virtual ~Universe() {
            for (auto* g : galaxies) delete g;
        }

        void addGalaxy(Galaxy* g) { galaxies.push_back(g); }
        size_t getGalaxyCount() const { return galaxies.size(); }
        int getDensityCount() const { return densityCount; }
    };
}