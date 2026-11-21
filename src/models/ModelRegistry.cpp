/**
 * @file ModelRegistry.cpp
 * @brief Implementation of the domain-filtered three-tier `ModelRegistry`.
 */

#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/models/FieldGrid2DSolver.hpp"
#include "umgebung/models/RadialScaleProfileSolver.hpp"
#include "umgebung/models/mainstream/DarkMatterNFWSolver.hpp"
#include "umgebung/models/mainstream/NewtonianNBodySolver.hpp"
#include "umgebung/models/thinkers/DanaanSolver.hpp"
#include "umgebung/models/thinkers/EssonneSolver.hpp"
#include "umgebung/models/thinkers/HarameinSolver.hpp"
#include "umgebung/models/thinkers/PaisSolver.hpp"
#include "umgebung/models/thinkers/TitorC204Solver.hpp"
#include "umgebung/models/thinkers/VogelWillisSolver.hpp"
#include "umgebung/models/unified/UnifiedSynthesisSolver.hpp"

namespace umgebung::models {

    const std::vector<ModelDescriptor>& ModelRegistry::GetAllModels() noexcept {
        using namespace scene;
        static const std::vector<ModelDescriptor> kModels = {
            // Tier 1: Mainstream Physics Baselines
            {
                ModelId::MainstreamNewtonian,
                "Mainstream::Newtonian",
                "Mainstream: Newtonian N-Body (1/r^2)",
                "Tier 1: Mainstream",
                "Classical Newtonian Mechanics + Symplectic Velocity Verlet",
                kDomainNBodyDynamics,
                ModelProvenance::TextbookBaseline,
                "Textbook pairwise 1/r^2 gravity & symplectic Velocity Verlet (formally proven in Lean 4)."
            },
            {
                ModelId::MainstreamDarkMatterNFW,
                "Mainstream::DarkMatterNFW",
                "Mainstream: Dark Matter NFW Halo",
                "Tier 1: Mainstream",
                "Navarro, Frenk & White (ApJ 1996 / 1997) CDM Halo Baseline",
                kDomainNBodyDynamics,
                ModelProvenance::TextbookBaseline,
                "Textbook NFW halo density, enclosed mass M_NFW(r), and conservative potential Phi_NFW(r)."
            },
            {
                ModelId::MainstreamMaxwellEM,
                "Mainstream::MaxwellEM",
                "Mainstream: Maxwell 2D FDTD EM Cavity",
                "Tier 1: Mainstream",
                "Classical Maxwell TMz Yee-Lattice FDTD Electrodynamics",
                kDomainFieldGrid2D,
                ModelProvenance::TextbookBaseline,
                "Textbook 2D TMz Yee-grid Maxwell FDTD with CFL bound S <= 0.45 < 1/sqrt(2) (proven in Lean 4)."
            },
            {
                ModelId::MainstreamSchwarzschildGR,
                "Mainstream::SchwarzschildGR",
                "Mainstream: Schwarzschild GR Horizon & Mass",
                "Tier 1: Mainstream",
                "General Relativity Schwarzschild Metric g_tt = 1 - R_s/r",
                kDomainRadialAndScaleProfile,
                ModelProvenance::TextbookBaseline,
                "Textbook Schwarzschild metric g_tt = 1 - R_s/r, M_s(r) = c^2 r / (2G), and horizon r = R_s."
            },

            // Tier 2: Standalone Verbatim Thinker Modules
            {
                ModelId::ThinkerEssonne,
                "Thinker::Essonne",
                "Thinker: Chris Essonne (Vacuum Gradient & 2D Vortex)",
                "Tier 2: Thinker",
                "Chris Essonne (8052.pdf & 10100.pdf Eq. 7, 11, 21, 27, 35, 43)",
                kDomainNBodyDynamics | kDomainFieldGrid2D | kDomainRadialAndScaleProfile,
                ModelProvenance::VerbatimEquation,
                "Verbatim: 8052.pdf Eq. 21/23/27/35/39/43 (P/P0, c', R=3/2 Rs, r=2 Rs) & 10100.pdf Eq. 7/11 2D vortex."
            },
            {
                ModelId::ThinkerHaramein,
                "Thinker::Haramein",
                "Thinker: Nassim Haramein (Holographic PSU & Torque)",
                "Tier 2: Thinker",
                "Nassim Haramein (Haramein342013PRRI3363.pdf & Scale-Unification)",
                kDomainNBodyDynamics | kDomainRadialAndScaleProfile,
                ModelProvenance::VerbatimEquation,
                "Verbatim: Haramein 2013 Eq. 5-13 & 24-27 (PSU eta, R, phi, m_h, m_p') + Haramein-Rauscher Coriolis torque."
            },
            {
                ModelId::ThinkerPais,
                "Thinker::Pais",
                "Thinker: Salvatore Pais (HEEMFG Resonant Cavity)",
                "Tier 2: Thinker",
                "Salvatore Pais (US Patent 10,144,532 B2 Eq. 1 & 2)",
                kDomainFieldGrid2D,
                ModelProvenance::HybridVerbatimAndAnsatz,
                "Verbatim: US10144532 Eq. 1-2 S_max | Ansatz: vacuum-polarization factor 1/(1 + kappa * S_max) in 2D FDTD."
            },
            {
                ModelId::ThinkerDanaan,
                "Thinker::Danaan",
                "Thinker: Elena Danaan (Love Coherence & Non-Linear Time)",
                "Tier 2: Thinker",
                "Elena Danaan Cosmology (Flower of Life, Love & Vibrational Density)",
                kDomainNBodyDynamics,
                ModelProvenance::OperationalizedSpec,
                "Verbatim: Flower of Life lattice & Void | Ansatz: antisymmetric Love pair force & dTau/dt = sqrt(P/P0)*(D/D0)."
            },
            {
                ModelId::ThinkerVogelWillis,
                "Thinker::VogelWillis",
                "Thinker: Vogel / Willis / Eredyon (4096 Hz Phryll Crystal)",
                "Tier 2: Thinker",
                "Dan Willis & Marcel Vogel (PhryllGenInstructions.pdf 51.84/60 deg)",
                kDomainFieldGrid2D,
                ModelProvenance::HybridVerbatimAndAnsatz,
                "Verbatim: 51.84/60 deg cuts, 4096 Hz 180-deg coils, 20-60V safe window | Ansatz: 2D FDTD drive power & saturation."
            },
            {
                ModelId::ThinkerTitorC204,
                "Thinker::TitorC204",
                "Thinker: John Titor (GE C204 Gravity Distortion Time Displacement Unit)",
                "Tier 2: Thinker",
                "John Titor C204 Manual (TMIM-200S/4005), Pohlman US20060073976A1 & Tipler (1974)",
                kDomainFieldGrid2D | kDomainRadialAndScaleProfile,
                ModelProvenance::HybridVerbatimAndAnsatz,
                "Verbatim: C204 Manual (22x6x6m, 5m offset, 50/75/100% power, 126k/201.6k/288k C, 3.4/6.2/10 yrs/hr, 240s VGL, 170-deg X-ray cone) & Tipler/Kerr-Newman metrics | Ansatz: Calibrated 3+1 ADM Van Den Broeck phenomenological metric g_00^eff."
            },

            // Tier 3: Combined Unified Synthesis
            {
                ModelId::UnifiedSynthesis,
                "Unified::Synthesis",
                "Unified: Full Synthesis (All Thinkers Composed)",
                "Tier 3: Unified",
                "Composed Essonne + Haramein + Pais + Danaan + Vogel/Willis",
                kDomainNBodyDynamics | kDomainFieldGrid2D,
                ModelProvenance::HybridVerbatimAndAnsatz,
                "Composes all verbatim Thinker equations & operationalized ansatzes into one multi-scale GPU solver."
            }
        };
        return kModels;
    }

    std::vector<ModelDescriptor> ModelRegistry::GetModelsForDomain(scene::SimulationDomain domain) {
        std::vector<ModelDescriptor> filtered;
        for (const auto& desc : GetAllModels()) {
            if (scene::MaskSupportsDomain(desc.supportedDomains, domain)) {
                filtered.push_back(desc);
            }
        }
        return filtered;
    }

    bool ModelRegistry::IsModelCompatibleWithDomain(
        scene::ModelId id,
        scene::SimulationDomain domain
    ) noexcept {
        const ModelDescriptor* desc = FindById(id);
        return (desc != nullptr) && scene::MaskSupportsDomain(desc->supportedDomains, domain);
    }

    scene::ModelId ModelRegistry::GetDefaultModelForDomain(scene::SimulationDomain domain) noexcept {
        switch (domain) {
            case scene::SimulationDomain::NBodyDynamics:         return scene::ModelId::MainstreamNewtonian;
            case scene::SimulationDomain::FieldGrid2D:           return scene::ModelId::MainstreamMaxwellEM;
            case scene::SimulationDomain::RadialAndScaleProfile: return scene::ModelId::MainstreamSchwarzschildGR;
        }
        return scene::ModelId::MainstreamNewtonian;
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
        scene::SimulationDomain domain,
        std::size_t streamIndex
    ) {
        if (!IsModelCompatibleWithDomain(id, domain)) {
            return nullptr;
        }

        std::unique_ptr<IPhysicsSolver> solver;
        if (domain == scene::SimulationDomain::FieldGrid2D) {
            if (id == scene::ModelId::ThinkerTitorC204) {
                solver = std::make_unique<thinkers::TitorC204Solver>();
            } else {
                solver = std::make_unique<FieldGrid2DSolver>(id);
            }
        } else if (domain == scene::SimulationDomain::RadialAndScaleProfile) {
            solver = std::make_unique<RadialScaleProfileSolver>(id);
        } else {
            // SimulationDomain::NBodyDynamics
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
                case scene::ModelId::ThinkerDanaan:
                    solver = std::make_unique<thinkers::DanaanSolver>();
                    break;
                case scene::ModelId::UnifiedSynthesis:
                    solver = std::make_unique<unified::UnifiedSynthesisSolver>();
                    break;
                default:
                    return nullptr;
            }
        }

        if (solver) {
            solver->SetStreamIndex(streamIndex);
        }
        return solver;
    }

    std::unique_ptr<IPhysicsSolver> ModelRegistry::CreateSolver(
        scene::ModelId id,
        std::size_t streamIndex
    ) {
        if (IsModelCompatibleWithDomain(id, scene::SimulationDomain::NBodyDynamics)) {
            return CreateSolver(id, scene::SimulationDomain::NBodyDynamics, streamIndex);
        }
        if (IsModelCompatibleWithDomain(id, scene::SimulationDomain::FieldGrid2D)) {
            return CreateSolver(id, scene::SimulationDomain::FieldGrid2D, streamIndex);
        }
        return CreateSolver(id, scene::SimulationDomain::RadialAndScaleProfile, streamIndex);
    }

} // namespace umgebung::models
