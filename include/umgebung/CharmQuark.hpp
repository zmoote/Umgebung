#pragma once
#include "Quark.hpp"
#include "Constants.hpp"

namespace Umgebung {
    class CharmQuark : public Quark {
    public:
        CharmQuark()
            : Quark(ParticleType::Matter,
                MeVToKg(1275.0),           // 1275 MeV/c²
                2.0 / 3.0 * elementary_charge, // +2/3 e
                QuarkFlavor::Charm,
                "red") {
        }
        std::string getParticleName() const override { return "CharmQuark"; }
    };
}