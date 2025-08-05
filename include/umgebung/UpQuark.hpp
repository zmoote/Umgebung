#pragma once
#include "Quark.hpp"

namespace Umgebung {
    class UpQuark : public Quark {
    public:
        UpQuark() : Quark(ParticleType::Matter, 2.3e-30, 2.0 / 3.0 * 1.6e-19, QuarkFlavor::Up, "red") {}
        std::string getParticleName() const override { return "UpQuark"; }
    };
}