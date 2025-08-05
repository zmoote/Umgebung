#pragma once
#include "SubatomicParticle.hpp"
#include "Constants.hpp"
#include <string>

namespace Umgebung {
    enum class BosonType { Photon, Gluon, W, Z, Higgs };

    class Boson : public SubatomicParticle {
    protected:
        BosonType bosonType;
    public:
        Boson(ParticleType t, double m, double c, BosonType bt)
            : SubatomicParticle(t, m, c, 1.0), bosonType(bt) {
        } // Default spin 1 (except Higgs: 0)
        virtual ~Boson() = default;

        double getMass() const override { return mass; }
        double getCharge() const override { return charge; }
        ParticleType getParticleType() const override { return type; }
        double getSpin() const override { return spin; }
        std::string getParticleName() const override { return "Boson"; }

        BosonType getBosonType() const { return bosonType; }
        void setBosonType(BosonType bt) { bosonType = bt; }
    };
}