#pragma once
#include "Boson.hpp"
#include "Constants.hpp"

namespace Umgebung {
    class Photon : public Boson {
    public:
        Photon()
            : Boson(ParticleType::Matter, 0.0, 0.0, BosonType::Photon) {
        } // Massless, neutral
        std::string getParticleName() const override { return "Photon"; }
    };
}