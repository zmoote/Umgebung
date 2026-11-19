/**
 * @file ModelRegistry.cpp
 * @brief Implementation of the three-tier `umgebung::models::ModelRegistry`.
 */

#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/models/mainstream/DarkMatterNFWSolver.hpp"
#include "umgebung/models/mainstream/NewtonianNBodySolver.hpp"
#include "umgebung/models/thinkers/DanaanSolver.hpp"
#include "umgebung/models/thinkers/EssonneSolver.hpp"
#include "umgebung/models/thinkers/HarameinSolver.hpp"
#include "umgebung/models/thinkers/PaisSolver.hpp"
#include "umgebung/models/thinkers/VogelWillisSolver.hpp"
#include "umgebung/models/unified/UnifiedSynthesisSolver.hpp"

namespace umgebung::models {

    const std::vector<ModelDescriptor>& ModelRegistry::GetAllModels() noexcept {
        static const std::vector<ModelDescriptor> kModels = {
            // Tier 1: Mainstream Physics Baselines
            {
                scene::ModelId::MainstreamNewtonian,
                "Mainstream::Newtonian",
                "Mainstream: Newtonian N-Body (1/r^2)",
                "Tier 1: Mainstream",
                "Classical Newtonian Mechanics + Symplectic Velocity Verlet"
            },
            {
                scene::ModelId::MainstreamDarkMatterNFW,
                "Mainstream::DarkMatterNFW",
                "Mainstream: Dark Matter NFW Halo",
                "Tier 1: Mainstream",
                "Navarro, Frenk & White (ApJ 1996 / 1997) CDM Halo Baseline"
            },

            // Tier 2: Standalone Verbatim Thinker Modules
            {
                scene::ModelId::ThinkerEssonne,
                "Thinker::Essonne",
                "Thinker: Chris Essonne (Vacuum Gradient & 2D Vortex)",
                "Tier 2: Thinker",
                "Chris Essonne (8052.pdf & 10100.pdf Eq. 7, 11, 21, 27)"
            },
            {
                scene::ModelId::ThinkerHaramein,
                "Thinker::Haramein",
                "Thinker: Nassim Haramein (Holographic PSU & Torque)",
                "Tier 2: Thinker",
                "Nassim Haramein (Haramein342013PRRI3363.pdf & Scale-Unification)"
            },
            {
                scene::ModelId::ThinkerPais,
                "Thinker::Pais",
                "Thinker: Salvatore Pais (HEEMFG Inertial Reduction)",
                "Tier 2: Thinker",
                "Salvatore Pais (US Patent 10,144,532 B2 Eq. 1 & 2)"
            },
            {
                scene::ModelId::ThinkerDanaan,
                "Thinker::Danaan",
                "Thinker: Elena Danaan (Love Coherence & Non-Linear Time)",
                "Tier 2: Thinker",
                "Elena Danaan Cosmology (Flower of Life, Love & Vibrational Density)"
            },
            {
                scene::ModelId::ThinkerVogelWillis,
                "Thinker::VogelWillis",
                "Thinker: Vogel / Willis / Eredyon (4096 Hz Phryll Crystal)",
                "Tier 2: Thinker",
                "Dan Willis & Marcel Vogel (PhryllGenInstructions.pdf 51.84/60 deg)"
            },

            // Tier 3: Combined Unified Synthesis
            {
                scene::ModelId::UnifiedSynthesis,
                "Unified::Synthesis",
                "Unified: Full Synthesis (All 5 Thinkers Composed)",
                "Tier 3: Unified",
                "Composed Essonne + Haramein + Pais + Danaan + Vogel/Willis"
            }
        };
        return kModels;
    }

    const ModelDescriptor* ModelRegistry::FindById(scene::ModelId id) noexcept {
        for (const auto& desc : GetAllModels()) {
            if (desc.id == id) {
                return &desc;
            }
        }
        return nullptr;
    }

    const ModelDescriptor* ModelRegistry::FindByName(const std::string& name) noexcept {
        scene::ModelId parsed{};
        if (!scene::ModelIdFromString(name, parsed)) {
            return nullptr;
        }
        return FindById(parsed);
    }

    std::unique_ptr<IPhysicsSolver> ModelRegistry::CreateSolver(
        scene::ModelId id,
        std::size_t streamIndex
    ) {
        std::unique_ptr<IPhysicsSolver> solver;
        switch (id) {
            case scene::ModelId::MainstreamNewtonian:
                solver = std::make_unique<mainstream::NewtonianNBodySolver>();
                break;
            case scene::ModelId::MainstreamDarkMatterNFW:
                solver = std::make_unique<mainstream::DarkMatterNFWSolver>();
                break;
            case scene::ModelId::ThinkerEssonne:
                solver = std::make_unique<thinkers::EssonneSolver>();
                break;
            case scene::ModelId::ThinkerHaramein:
                solver = std::make_unique<thinkers::HarameinSolver>();
                break;
            case scene::ModelId::ThinkerPais:
                solver = std::make_unique<thinkers::PaisSolver>();
                break;
            case scene::ModelId::ThinkerDanaan:
                solver = std::make_unique<thinkers::DanaanSolver>();
                break;
            case scene::ModelId::ThinkerVogelWillis:
                solver = std::make_unique<thinkers::VogelWillisSolver>();
                break;
            case scene::ModelId::UnifiedSynthesis:
                solver = std::make_unique<unified::UnifiedSynthesisSolver>();
                break;
        }

        if (!solver) {
            solver = std::make_unique<mainstream::NewtonianNBodySolver>();
        }
        solver->SetStreamIndex(streamIndex);
        return solver;
    }

} // namespace umgebung::models
