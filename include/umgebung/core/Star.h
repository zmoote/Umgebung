#pragma once
#include "CelestialBody.h"

namespace Umgebung {
    class Star : public CelestialBody {
    public:
        Star(float mass, float radius);

        void computeCustomPhysics(float dt) override;   // Holographic mass + ether pressure
    };
}