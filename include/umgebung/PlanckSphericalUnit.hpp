#pragma once
#include "QuantumVacuumFluctuation.hpp"

namespace Umgebung {
    class PlanckSphericalUnit : public QuantumVacuumFluctuation {
    protected:
        double radius; // Planck length scale
    public:
        PlanckSphericalUnit(double f, double r)
            : QuantumVacuumFluctuation(f), radius(r) {
        }

        double getFrequency() const override { return frequency; }
        double getRadius() const { return radius; }
        void setRadius(double r) { radius = r; }
    };
}