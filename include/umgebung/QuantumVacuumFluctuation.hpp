#pragma once

namespace Umgebung {
    class QuantumVacuumFluctuation : public ElementaryParticle {
    protected:
        double frequency; // Oscillation frequency (Hz)
    public:
        QuantumVacuumFluctuation(double f) : frequency(f) {}
        virtual ~QuantumVacuumFluctuation() = default;

        virtual double getFrequency() const = 0;
        void setFrequency(double f) { frequency = f; }
    };
}