#pragma once
#include "Lepton.hpp"

namespace Umgebung {
    class Positron : public Lepton {
    public:
        Positron() : Lepton(ParticleType::Antimatter, 9.11e-31, 1.6e-19, LeptonType::Electron) {}
        std::string getParticleName() const override { return "Positron"; }
    };
}