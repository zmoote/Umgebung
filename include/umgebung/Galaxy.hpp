#pragma once
#include "SolarSystem.hpp"
#include <vector>

namespace Umgebung {
    class Galaxy {
    protected:
        std::vector<SolarSystem*> solarSystems;
    public:
        Galaxy() = default;
        virtual ~Galaxy() {
            for (auto* s : solarSystems) delete s;
        }

        void addSolarSystem(SolarSystem* s) { solarSystems.push_back(s); }
        size_t getSolarSystemCount() const { return solarSystems.size(); }
    };
}