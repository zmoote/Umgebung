#pragma once
#include "Quark.hpp"
#include "Constants.hpp"

namespace Umgebung {
    class UpQuark : public Quark {
    public:
        UpQuark() : Quark(ParticleType::Matter, MeVToKg(2.2), 2.0 / 3.0 * elementary_charge, QuarkFlavor::Up, "red") {}
        std::string getParticleName() const override { return "UpQuark"; }
    };
}