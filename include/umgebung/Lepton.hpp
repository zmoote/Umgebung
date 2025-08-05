#pragma once
#include "SubatomicParticle.hpp"

namespace Umgebung {
    enum class LeptonType { Electron, Muon, Tau, ElectronNeutrino, MuonNeutrino, TauNeutrino };

    class Lepton : public SubatomicParticle {
    protected:
        LeptonType leptonType;
    public:
        Lepton(ParticleType t, double m, double c, LeptonType lt)
            : SubatomicParticle(t, m, c, 0.5), leptonType(lt) {
        }
        virtual ~Lepton() = default;

        double getMass() const override { return mass; }
        double getCharge() const override { return charge; }
        ParticleType getParticleType() const override { return type; }
        double getSpin() const override { return spin; }
        std::string getParticleName() const override { return "Lepton"; }

        LeptonType getLeptonType() const { return leptonType; }
        void setLeptonType(LeptonType lt) { leptonType = lt; }
    };
}