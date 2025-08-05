#pragma once
#include "Ecosystem.hpp"
#include <vector>

namespace Umgebung {
    class SurfaceFeature {
    protected:
        std::vector<Ecosystem*> ecosystems;
    public:
        SurfaceFeature() = default;
        virtual ~SurfaceFeature() {
            for (auto* e : ecosystems) delete e;
        }

        void addEcosystem(Ecosystem* e) { ecosystems.push_back(e); }
        size_t getEcosystemCount() const { return ecosystems.size(); }
    };
}