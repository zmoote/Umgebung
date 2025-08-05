#pragma once
#include "SubatomicParticle.hpp"

namespace Umgebung {
    enum class QuarkFlavor { Up, Down, Charm, Strange, Top, Bottom };

    class Quark : public SubatomicParticle {
    protected:
        QuarkFlavor flavor;
        std::string color; // "red", "green", "blue"
    public:
        Quark(ParticleType t, double m, double c, QuarkFlavor f, const std::string& col)
            : SubatomicParticle(t, m, c, 0.5), flavor(f), color(col) {
        }
        virtual ~Quark() = default;

        double getMass() const override { return mass; }
        double getCharge() const override { return charge; }
        ParticleType getParticleType() const override { return type; }
        double getSpin() const override { return spin; }
        std::string getParticleName() const override { return "Quark"; }

        QuarkFlavor getFlavor() const { return flavor; }
        std::string getColor() const { return color; }
        void setFlavor(QuarkFlavor f) { flavor = f; }
        void setColor(const std::string& c) { color = c; }
    };
}