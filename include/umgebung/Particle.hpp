#pragma once

namespace Umgebung {
    enum class ParticleType {
        Matter,
        Antimatter
    };

    class Particle {
    protected:
        ParticleType type;
        double mass;   // kg, allows negative for edge cases
        double charge; // Coulombs
    public:
        Particle(ParticleType t, double m, double c)
            : type(t), mass(m), charge(c) {
        }
        virtual ~Particle() = default;

        // Pure virtual getters
        virtual double getMass() const = 0;
        virtual double getCharge() const = 0;
        virtual ParticleType getParticleType() const = 0;

        // Setters for theoretical flexibility
        void setMass(double m) { mass = m; }
        void setCharge(double c) { charge = c; }
        void setType(ParticleType t) { type = t; }
    };
}