#pragma once
#include "Particle.hpp"
#include <vector>
#include <string>

namespace Umgebung {
    class ElementaryParticle;

    class SubatomicParticle : public Particle {
    protected:
        double spin;
        std::vector<ElementaryParticle*> constituents;
    public:
        SubatomicParticle(ParticleType t, double m, double c, double s)
            : Particle(t, m, c), spin(s) {
        }
        virtual ~SubatomicParticle() {
            for (auto* p : constituents) delete p;
        }

        virtual double getSpin() const = 0;
        virtual std::string getParticleName() const = 0;
        void setSpin(double s) { spin = s; }
        void addConstituent(ElementaryParticle* p) { constituents.push_back(p); }
    };
}