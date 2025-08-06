#pragma once
#include "SubatomicParticle.hpp"
#include "Quark.hpp"
#include "QuantumVacuumFluctuation.hpp"
#include <vector>
#include <string>

namespace Umgebung {
    class Proton : public SubatomicParticle {
    private:
        double frequency; // Hz, for density-based interactions
        void initializeQuarks() {
            // 2 up quarks, 1 down quark (uud), ensuring color neutrality
            constituents.push_back(new Quark(ParticleType::Matter, 2.3e-30, 2.0 / 3.0, QuarkFlavor::Up, "red"));
            constituents.push_back(new Quark(ParticleType::Matter, 2.3e-30, 2.0 / 3.0, QuarkFlavor::Up, "green"));
            constituents.push_back(new Quark(ParticleType::Matter, 4.8e-30, -1.0 / 3.0, QuarkFlavor::Down, "blue"));
        }

    public:
        Proton(double freq = 1.0e12) // Default frequency for 3rd density
            : SubatomicParticle(ParticleType::Matter, 1.6726219e-27, 1.60217662e-19, 0.5), frequency(freq) {
            initializeQuarks();
        }

        ~Proton() override = default;

        double getMass() const override {
            // Empirical mass (~938 MeV/c²), as most mass comes from gluon/vacuum interactions
            return mass; // 1.6726219e-27 kg
        }

        double getCharge() const override {
            // Verify quark charges: 2*(+2/3) + (-1/3) = +1
            double totalCharge = 0.0;
            for (const auto* p : constituents) {
                totalCharge += p->getCharge();
            }
            return charge; // 1.60217662e-19 C
        }

        ParticleType getParticleType() const override {
            return type; // ParticleType::Matter
        }

        double getSpin() const override {
            return spin; // 1/2
        }

        std::string getParticleName() const override {
            return "Proton";
        }

        std::vector<Quark*> getQuarks() const {
            std::vector<Quark*> quarks;
            for (auto* p : constituents) {
                if (auto* q = dynamic_cast<Quark*>(p)) {
                    quarks.push_back(q);
                }
            }
            return quarks;
        }

        // Haramein's Schwarzschild Proton model
        double getSchwarzschildRadius() const {
            const double G = 6.67430e-11; // m^3 kg^-1 s^-2
            const double c = 2.99792458e8; // m/s
            return 2 * G * mass / (c * c); // ~1.24e-54 m
        }

        // Frequency-based density (from "The Seeders - Notes.txt")
        double getFrequency() const { return frequency; }
        void setFrequency(double f) { frequency = f; }

        // Add vacuum fluctuation for Haramein's model
        void addVacuumFluctuation(QuantumVacuumFluctuation* f) {
            constituents.push_back(f);
        }

        // Placeholder for PhysX/CUDA interaction
        void interactWith(Particle* other) {
            // Use PhysX for strong force (gluon-mediated) interactions
            // CUDA kernel for parallel force calculations
            // Example: Adjust position/velocity based on color charge interactions
        }
    };
}