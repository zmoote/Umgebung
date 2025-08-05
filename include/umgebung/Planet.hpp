#pragma once
#include "SurfaceFeature.hpp"
#include <vector>

namespace Umgebung {
    class Planet {
    protected:
        std::vector<SurfaceFeature*> features;
    public:
        Planet() = default;
        virtual ~Planet() {
            for (auto* f : features) delete f;
        }

        void addSurfaceFeature(SurfaceFeature* f) { features.push_back(f); }
        size_t getFeatureCount() const { return features.size(); }
    };
}