#pragma once
#include "QuantumVacuumFluctuation.hpp"
#include <vector>

namespace Umgebung {
    class ElementaryParticle {
    protected:
        std::vector<QuantumVacuumFluctuation*> fluctuations; // Optional composition
        double mass; // kg, allows negative
    public:
        ElementaryParticle(double m) : mass(m) {}
        virtual ~ElementaryParticle() {
            for (auto* f : fluctuations) delete f;
        }

        virtual double getMass() const = 0;
        void setMass(double m) { mass = m; }
        void addFluctuation(QuantumVacuumFluctuation* f) { fluctuations.push_back(f); }
    };
}