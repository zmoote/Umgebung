/**
 * @file StandaloneThinkersAndNFWTests.cpp
 * @brief Automated CTest verification suite covering:
 *   1. `ModelRegistry` enumeration & lookup of all 8 three-tier models.
 *   2. `Mainstream::DarkMatterNFW`: analytical NFW density/mass/potential checks
 *      + live GPU 3-way spiral galaxy comparison (`Newtonian` vs `DarkMatterNFW`
 *      vs `Thinker::Essonne`) on a single `galaxy_periphery.scenario.umgebung`.
 *   3. Standalone `Thinker::Essonne`: exact flat circular orbits & energy conservation.
 *   4. Standalone `Thinker::Haramein`: Strang-split Coriolis spacetime torque
 *      precession with strict energy conservation (`v . a_Cor = 0`) & PSU `phi`.
 *   5. Standalone `Thinker::Pais`: HEEMFG Eq. (1) Poynting flux, inertial mass
 *      reduction `m_eff / m_0 < 1`, and measurable orbital divergence vs Newtonian.
 *   6. Standalone `Thinker::Danaan`: Love binding momentum/energy conservation
 *      and vibrational-density non-linear proper-time divergence.
 *   7. Standalone `Thinker::VogelWillis`: 51.84/60 deg resonance, 4096 Hz coil
 *      power, Phryll momentum/energy conservation, and crystal singing trigger.
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/models/mainstream/DarkMatterNFWSolver.hpp"
#include "umgebung/models/thinkers/VogelWillisModel.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int gFailedAssertions = 0;

    void ExpectTrue(bool condition, const char* exprText, const char* testName) {
        if (!condition) {
            std::cerr << "[FAIL] " << testName << ": assertion failed -> " << exprText << std::endl;
            ++gFailedAssertions;
        }
    }

    void TestModelRegistryCompleteness() {
        const char* testName = "TestModelRegistryCompleteness";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        const auto& all = ModelRegistry::GetAllModels();
        ExpectTrue(all.size() == 8U, "ModelRegistry registers all 8 three-tier models", testName);
        ExpectTrue(ModelRegistry::FindByName("Mainstream::Newtonian") != nullptr, "Find Mainstream::Newtonian", testName);
        ExpectTrue(ModelRegistry::FindByName("Mainstream::DarkMatterNFW") != nullptr, "Find Mainstream::DarkMatterNFW", testName);
        ExpectTrue(ModelRegistry::FindByName("Thinker::Essonne") != nullptr, "Find Thinker::Essonne", testName);
        ExpectTrue(ModelRegistry::FindByName("Thinker::Haramein") != nullptr, "Find Thinker::Haramein", testName);
        ExpectTrue(ModelRegistry::FindByName("Thinker::Pais") != nullptr, "Find Thinker::Pais", testName);
        ExpectTrue(ModelRegistry::FindByName("Thinker::Danaan") != nullptr, "Find Thinker::Danaan", testName);
        ExpectTrue(ModelRegistry::FindByName("Thinker::VogelWillis") != nullptr, "Find Thinker::VogelWillis", testName);
        ExpectTrue(ModelRegistry::FindByName("Unified::Synthesis") != nullptr, "Find Unified::Synthesis", testName);
    }

    void TestDarkMatterNFWAndThreeWayGalaxy(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestDarkMatterNFWAndThreeWayGalaxy";
        using namespace umgebung::models;
        using namespace umgebung::models::mainstream;
        using namespace umgebung::scene;

        // Analytical NFW checks (matches formal/Umgebung/Mainstream/DarkMatterNFW.lean).
        const double rhoAtRs = ComputeNFWDensity(3.0, 0.04, 3.0);
        ExpectTrue(std::abs(rhoAtRs - 0.01) < 1e-15, "NFW rho(r_s) == rho0 / 4", testName);
        const double phi0 = ComputeNFWPotential(0.0, 1.0, 0.03, 3.0);
        const double phiNear0 = ComputeNFWPotential(1.0e-8, 1.0, 0.03, 3.0);
        ExpectTrue(std::abs(phiNear0 - phi0) / std::abs(phi0) < 1e-7, "NFW potential continuous at r -> 0", testName);

        // Verify numerical derivative dPhi/dr == +g_NFW(r).
        const double rTest = 2.5;
        const double eps = 1.0e-6;
        const double dPhiDr = (ComputeNFWPotential(rTest + eps, 1.0, 0.030947, 3.0) -
                               ComputeNFWPotential(rTest - eps, 1.0, 0.030947, 3.0)) / (2.0 * eps);
        const double gNfw = ComputeNFWRadialAccel(rTest, 1.0, 0.030947, 3.0);
        ExpectTrue(std::abs(dPhiDr - gNfw) / gNfw < 1e-8, "dPhi_NFW/dr == g_NFW(r) to < 1e-8", testName);

        // Live GPU 3-way galaxy periphery comparison on ONE scenario file!
        UniverseScene galaxy{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "galaxy_periphery.scenario.umgebung", galaxy, err),
            "Load galaxy_periphery.scenario.umgebung",
            testName
        );

        auto sNewton = ModelRegistry::CreateSolver(ModelId::MainstreamNewtonian, 0U);
        auto sNfw = ModelRegistry::CreateSolver(ModelId::MainstreamDarkMatterNFW, 1U);
        auto sEssonne = ModelRegistry::CreateSolver(ModelId::ThinkerEssonne, 0U);

        ExpectTrue(sNewton->Initialize(ctx, galaxy), "Init Newtonian", testName);
        ExpectTrue(sNfw->Initialize(ctx, galaxy), "Init DarkMatterNFW", testName);
        ExpectTrue(sEssonne->Initialize(ctx, galaxy), "Init Thinker::Essonne", testName);

        ExpectTrue(sNewton->StepMultiple(2000ULL, 0.002), "Step Newtonian", testName);
        ExpectTrue(sNfw->StepMultiple(2000ULL, 0.002), "Step DarkMatterNFW", testName);
        ExpectTrue(sEssonne->StepMultiple(2000ULL, 0.002), "Step Thinker::Essonne", testName);

        SolverDiagnostics dNfw{}, dEss{};
        ExpectTrue(sNfw->ComputeSolverDiagnostics(dNfw), "NFW diagnostics", testName);
        ExpectTrue(sEssonne->ComputeSolverDiagnostics(dEss), "Essonne diagnostics", testName);

        std::vector<BodyEntity> bNewt = galaxy.bodies, bNfw = galaxy.bodies, bEss = galaxy.bodies;
        ExpectTrue(sNewton->SyncToHost(bNewt), "Sync Newtonian", testName);
        ExpectTrue(sNfw->SyncToHost(bNfw), "Sync NFW", testName);
        ExpectTrue(sEssonne->SyncToHost(bEss), "Sync Essonne", testName);

        const double rOuterNewt = (bNewt[3].position - bNewt[0].position).Norm();
        const double rOuterNfw = (bNfw[3].position - bNfw[0].position).Norm();
        const double rOuterEss = (bEss[3].position - bEss[0].position).Norm();

        std::cout << "[3-Way Galaxy Star 3 (r0=5.0)] Newtonian r=" << rOuterNewt
                  << " | NFW r=" << rOuterNfw
                  << " (drift=" << dNfw.relativeEnergyDrift << ")"
                  << " | Essonne r=" << rOuterEss
                  << " (drift=" << dEss.relativeEnergyDrift << ")" << std::endl;

        ExpectTrue(rOuterNewt > 6.0, "Newtonian outer star escapes > 6.0", testName);
        ExpectTrue(rOuterNfw > 4.2 && rOuterNfw < 5.4, "DarkMatterNFW keeps outer star tightly bound near r=5.0", testName);
        ExpectTrue(std::abs(rOuterEss - 5.0) < 1e-3, "Standalone Thinker::Essonne holds exact circular r=5.0", testName);
        ExpectTrue(dNfw.relativeEnergyDrift < 1e-6, "DarkMatterNFW symplectic energy conserved < 1e-6", testName);
        ExpectTrue(dNfw.nfwHaloPotentialEnergy < 0.0, "DarkMatterNFW reports negative halo potential U_NFW", testName);
        ExpectTrue(dEss.relativeEnergyDrift < 1e-6, "Standalone Thinker::Essonne energy conserved < 1e-6", testName);

        sNewton->Shutdown();
        sNfw->Shutdown();
        sEssonne->Shutdown();
    }

    void TestStandaloneHarameinSolver(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestStandaloneHarameinSolver";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        UniverseScene kepler{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "kepler_two_body.scenario.umgebung", kepler, err),
            "Load kepler_two_body.scenario.umgebung",
            testName
        );

        auto sNewt = ModelRegistry::CreateSolver(ModelId::MainstreamNewtonian, 0U);
        auto sHara = ModelRegistry::CreateSolver(ModelId::ThinkerHaramein, 1U);
        ExpectTrue(sNewt->Initialize(ctx, kepler), "Init Newtonian", testName);
        ExpectTrue(sHara->Initialize(ctx, kepler), "Init Thinker::Haramein", testName);

        ExpectTrue(sNewt->StepMultiple(2000ULL, 0.001), "Step Newtonian", testName);
        ExpectTrue(sHara->StepMultiple(2000ULL, 0.001), "Step Thinker::Haramein", testName);

        SolverDiagnostics dHara{};
        ExpectTrue(sHara->ComputeSolverDiagnostics(dHara), "Haramein diagnostics", testName);

        std::vector<BodyEntity> bNewt = kepler.bodies, bHara = kepler.bodies;
        ExpectTrue(sNewt->SyncToHost(bNewt) && sHara->SyncToHost(bHara), "Sync both", testName);

        const double deltaR = (bHara[1].position - bNewt[1].position).Norm();
        std::cout << "[Thinker::Haramein] relEnergyDrift=" << dHara.relativeEnergyDrift
                  << " | <phi>=" << dHara.meanHolographicPhi
                  << " | deltaR vs Newtonian=" << deltaR << std::endl;

        ExpectTrue(dHara.relativeEnergyDrift < 1e-6, "Strang-split Coriolis preserves energy < 1e-6", testName);
        ExpectTrue(dHara.meanHolographicPhi > 0.0, "Haramein holographic ratio phi > 0", testName);
        ExpectTrue(deltaR > 1e-2, "Spacetime torque Omega_z=0.02 produces measurable orbital precession", testName);

        sNewt->Shutdown();
        sHara->Shutdown();
    }

    void TestStandalonePaisSolver(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestStandalonePaisSolver";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        UniverseScene paisScene{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "pais_heemfg_cavity.scenario.umgebung", paisScene, err),
            "Load pais_heemfg_cavity.scenario.umgebung",
            testName
        );

        auto sNewt = ModelRegistry::CreateSolver(ModelId::MainstreamNewtonian, 0U);
        auto sPais = ModelRegistry::CreateSolver(ModelId::ThinkerPais, 1U);
        ExpectTrue(sNewt->Initialize(ctx, paisScene), "Init Newtonian", testName);
        ExpectTrue(sPais->Initialize(ctx, paisScene), "Init Thinker::Pais", testName);

        ExpectTrue(sNewt->StepMultiple(1500ULL, 0.001), "Step Newtonian", testName);
        ExpectTrue(sPais->StepMultiple(1500ULL, 0.001), "Step Thinker::Pais", testName);

        SolverDiagnostics dPais{};
        ExpectTrue(sPais->ComputeSolverDiagnostics(dPais), "Pais diagnostics", testName);

        std::vector<BodyEntity> bNewt = paisScene.bodies, bPais = paisScene.bodies;
        ExpectTrue(sNewt->SyncToHost(bNewt) && sPais->SyncToHost(bPais), "Sync both", testName);

        const double deltaR = (bPais[1].position - bNewt[1].position).Norm();
        std::cout << "[Thinker::Pais] S_max=" << dPais.paisMaxPoyntingFlux
                  << " | m_eff/m0=" << dPais.paisInertialMassFactor
                  << " | deltaR vs Newtonian=" << deltaR << std::endl;

        ExpectTrue(dPais.paisMaxPoyntingFlux > 0.0, "Pais HEEMFG S_max > 0", testName);
        ExpectTrue(dPais.paisInertialMassFactor > 0.0 && dPais.paisInertialMassFactor < 1.0, "Pais inertial mass reduced (0 < m_eff/m0 < 1)", testName);
        ExpectTrue(deltaR > 1e-2, "HEEMFG craft trajectory diverges measurably from Newtonian", testName);

        sNewt->Shutdown();
        sPais->Shutdown();
    }

    void TestStandaloneDanaanSolver(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestStandaloneDanaanSolver";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        UniverseScene erra{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "universes" / "erra_pleiades.umgebung", erra, err),
            "Load universes/erra_pleiades.umgebung",
            testName
        );

        auto sDan = ModelRegistry::CreateSolver(ModelId::ThinkerDanaan, 0U);
        ExpectTrue(sDan->Initialize(ctx, erra), "Init Thinker::Danaan", testName);

        SolverDiagnostics d0{}, df{};
        ExpectTrue(sDan->ComputeSolverDiagnostics(d0), "Danaan initial diagnostics", testName);
        ExpectTrue(sDan->StepMultiple(1500ULL, 0.001), "Step Thinker::Danaan", testName);
        ExpectTrue(sDan->ComputeSolverDiagnostics(df), "Danaan final diagnostics", testName);

        const double momDrift = (df.totalLinearMomentum - d0.totalLinearMomentum).Norm();
        std::vector<double> properTimes;
        ExpectTrue(sDan->GetLocalProperTimes(properTimes), "GetLocalProperTimes", testName);

        std::cout << "[Thinker::Danaan] relEnergyDrift=" << df.relativeEnergyDrift
                  << " | momDrift=" << momDrift
                  << " | <dTau/dt>=" << df.meanLocalTimeRate << std::endl;

        ExpectTrue(momDrift < 1e-12, "Danaan Love force conserves linear momentum < 1e-12", testName);
        ExpectTrue(df.relativeEnergyDrift < 1e-5, "Danaan symplectic energy conserved < 1e-5", testName);
        ExpectTrue(df.meanLocalTimeRate > 1.5, "5th-density Bubble Universe advances local proper time faster than 3rd-density baseline", testName);

        sDan->Shutdown();
    }

    void TestStandaloneVogelWillisSolver(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestStandaloneVogelWillisSolver";
        using namespace umgebung::models;
        using namespace umgebung::models::thinkers;
        using namespace umgebung::scene;

        ExpectTrue(IsVogelVoltageWithinSafeWindow(36.0), "36 V within [20, 60] V safe window", testName);
        ExpectTrue(!IsVogelVoltageWithinSafeWindow(75.0), "75 V outside safe window", testName);
        ExpectTrue(std::abs(ComputeVogelGeometricResonanceFactor(51.84, 60.0) - 1.0) < 1e-15, "Exact 51.84/60 deg resonance == 1.0", testName);

        UniverseScene vogel{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "vogel_phryll_crystal.scenario.umgebung", vogel, err),
            "Load vogel_phryll_crystal.scenario.umgebung",
            testName
        );
        // Lower saturation threshold slightly in test copy to verify singing alarm triggers cleanly.
        vogel.physicsProfile.unifiedField.vogelWillis.saturationThreshold = 20.0;

        auto sVogel = ModelRegistry::CreateSolver(ModelId::ThinkerVogelWillis, 0U);
        ExpectTrue(sVogel->Initialize(ctx, vogel), "Init Thinker::VogelWillis", testName);

        SolverDiagnostics d0{}, df{};
        ExpectTrue(sVogel->ComputeSolverDiagnostics(d0), "Vogel initial diagnostics", testName);
        ExpectTrue(!d0.vogelCrystalSinging, "Crystal not singing at t=0", testName);

        ExpectTrue(sVogel->StepMultiple(1000ULL, 0.002), "Step Thinker::VogelWillis", testName);
        ExpectTrue(sVogel->ComputeSolverDiagnostics(df), "Vogel final diagnostics", testName);

        const double momDrift = (df.totalLinearMomentum - d0.totalLinearMomentum).Norm();
        std::cout << "[Thinker::VogelWillis] relEnergyDrift=" << df.relativeEnergyDrift
                  << " | momDrift=" << momDrift
                  << " | satRatio=" << df.vogelCrystalSaturationRatio
                  << " | singing=" << (df.vogelCrystalSinging ? "YES" : "NO") << std::endl;

        ExpectTrue(momDrift < 1e-12, "Vogel/Willis Phryll force conserves linear momentum < 1e-12", testName);
        ExpectTrue(df.relativeEnergyDrift < 1e-5, "Vogel/Willis symplectic energy conserved < 1e-5", testName);
        ExpectTrue(df.vogelCrystalSaturationRatio >= 1.0 && df.vogelCrystalSinging, "Crystal saturation triggers singing flag", testName);

        sVogel->Shutdown();
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    TestModelRegistryCompleteness();

    const auto caps = umgebung::core::SystemProfiler::ProfileSystem();
    const auto plan = umgebung::core::SystemProfiler::BuildAdaptationPlan(
        caps,
        umgebung::core::HardwareAdaptationConfig{.guiActive = false}
    );

    umgebung::compute::CUDAContext ctx;
    if (!ctx.Initialize(caps, plan)) {
        std::cerr << "[FATAL] Failed to initialize CUDAContext for StandaloneThinkersAndNFWTests." << std::endl;
        return 1;
    }

    TestDarkMatterNFWAndThreeWayGalaxy(ctx, scenesDir);
    TestStandaloneHarameinSolver(ctx, scenesDir);
    TestStandalonePaisSolver(ctx, scenesDir);
    TestStandaloneDanaanSolver(ctx, scenesDir);
    TestStandaloneVogelWillisSolver(ctx, scenesDir);

    ctx.Shutdown();

    if (gFailedAssertions > 0) {
        std::cerr << "StandaloneThinkersAndNFWTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All StandaloneThinkersAndNFWTests passed." << std::endl;
    return 0;
}
