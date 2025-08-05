#pragma once
#include "Lepton.hpp"
#include "Constants.hpp"

namespace Umgebung {
    class Positron : public Lepton {
    public:
        Positron()
            : Lepton(ParticleType::Antimatter,
                MeVToKg(0.5109989461), // 0.5109989461 MeV/c²
                elementary_charge,     // +e
                LeptonType::Electron) {
        }
        std::string getParticleName() const override { return "Positron"; }
    };
}