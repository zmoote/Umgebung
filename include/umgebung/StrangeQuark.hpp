#pragma once
#include "Quark.hpp"
#include "Constants.hpp"

namespace Umgebung {
    class StrangeQuark : public Quark {
    public:
        StrangeQuark()
            : Quark(ParticleType::Matter,
                MeVToKg(93.0),             // 93 MeV/c²
                -1.0 / 3.0 * elementary_charge, // -1/3 e
                QuarkFlavor::Strange,
                "red") {
        }
        std::string getParticleName() const override { return "StrangeQuark"; }
    };
}