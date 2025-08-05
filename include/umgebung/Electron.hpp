#pragma once
#include "Lepton.hpp"

namespace Umgebung {
    class Electron : public Lepton {
    public:
        Electron() : Lepton(ParticleType::Matter, 9.11e-31, -1.6e-19, LeptonType::Electron) {}
        std::string getParticleName() const override { return "Electron"; }
    };
}