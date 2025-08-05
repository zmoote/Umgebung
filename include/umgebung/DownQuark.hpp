#pragma once
#include "Quark.hpp"
#include "Constants.hpp"

namespace Umgebung {
    class DownQuark : public Quark {
    public:
        DownQuark()
            : Quark(ParticleType::Matter,
                MeVToKg(4.7),              // 4.7 MeV/c²
                -1.0 / 3.0 * elementary_charge, // -1/3 e
                QuarkFlavor::Down,
                "red") {
        }
        std::string getParticleName() const override { return "DownQuark"; }
    };
}