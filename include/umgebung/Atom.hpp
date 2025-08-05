#pragma once
#include "SubatomicParticle.hpp"
#include <vector>

namespace Umgebung {
    class Atom {
    protected:
        std::vector<SubatomicParticle*> particles; // Protons, neutrons, electrons
    public:
        Atom() = default;
        virtual ~Atom() {
            for (auto* p : particles) delete p;
        }

        void addParticle(SubatomicParticle* p) { particles.push_back(p); }
        size_t getParticleCount() const { return particles.size(); }
    };
}