#pragma once

namespace Umgebung {
    class Star {
    protected:
        double mass; // kg
        double luminosity; // Watts
    public:
        Star(double m, double l) : mass(m), luminosity(l) {}
        virtual ~Star() = default;

        double getMass() const { return mass; }
        double getLuminosity() const { return luminosity; }
        void setMass(double m) { mass = m; }
        void setLuminosity(double l) { luminosity = l; }
    };
}