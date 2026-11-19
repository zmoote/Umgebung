/**
 * @file SimulationDomainTests.cpp
 * @brief Automated CTest verification suite for the `SimulationDomain`
 *        architecture: domain-filtered `ModelRegistry` guards, GPU
 *        `FieldGrid2DSolver` (CFL stability, MaxwellEM vs Pais vs VogelWillis
 *        vs Essonne), and `RadialScaleProfileSolver` (Haramein PSU scale sweep
 *        & Essonne vacuum horizon vs Mainstream::SchwarzschildGR).
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/models/thinkers/ThinkerConstants.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

    int gFailedAssertions = 0;

    void ExpectTrue(bool condition, const char* exprText, const char* testName) {
        if (!condition) {
            std::cerr << "[FAIL] " << testName << ": assertion failed -> " << exprText << std::endl;
            ++gFailedAssertions;
        }
    }

    /**
     * @brief Verifies strict SimulationDomain compatibility guards in ModelRegistry.
     */
    void TestDomainCompatibilityGuards() {
        const char* testName = "TestDomainCompatibilityGuards";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        // NBodyDynamics accepts orbital models and strictly rejects EM/crystal/1D-GR-only models.
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamNewtonian, SimulationDomain::NBodyDynamics), "Newtonian supports NBodyDynamics", testName);
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamDarkMatterNFW, SimulationDomain::NBodyDynamics), "NFW supports NBodyDynamics", testName);
        ExpectTrue(!ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerPais, SimulationDomain::NBodyDynamics), "Pais rejected for NBodyDynamics", testName);
        ExpectTrue(!ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerVogelWillis, SimulationDomain::NBodyDynamics), "VogelWillis rejected for NBodyDynamics", testName);
        ExpectTrue(!ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamMaxwellEM, SimulationDomain::NBodyDynamics), "MaxwellEM rejected for NBodyDynamics", testName);
        ExpectTrue(ModelRegistry::CreateSolver(ModelId::ThinkerVogelWillis, SimulationDomain::NBodyDynamics) == nullptr, "CreateSolver returns nullptr on incompatible domain", testName);

        // FieldGrid2D accepts MaxwellEM, Pais, VogelWillis, Essonne, Unified::Synthesis and rejects N-body only models.
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamMaxwellEM, SimulationDomain::FieldGrid2D), "MaxwellEM supports FieldGrid2D", testName);
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerPais, SimulationDomain::FieldGrid2D), "Pais supports FieldGrid2D", testName);
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerVogelWillis, SimulationDomain::FieldGrid2D), "VogelWillis supports FieldGrid2D", testName);
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerEssonne, SimulationDomain::FieldGrid2D), "Essonne supports FieldGrid2D", testName);
        ExpectTrue(!ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamNewtonian, SimulationDomain::FieldGrid2D), "Newtonian rejected for FieldGrid2D", testName);

        // RadialAndScaleProfile accepts SchwarzschildGR, Haramein, Essonne and rejects MaxwellEM/Pais.
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamSchwarzschildGR, SimulationDomain::RadialAndScaleProfile), "SchwarzschildGR supports RadialAndScaleProfile", testName);
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerHaramein, SimulationDomain::RadialAndScaleProfile), "Haramein supports RadialAndScaleProfile", testName);
        ExpectTrue(ModelRegistry::IsModelCompatibleWithDomain(ModelId::ThinkerEssonne, SimulationDomain::RadialAndScaleProfile), "Essonne supports RadialAndScaleProfile", testName);
        ExpectTrue(!ModelRegistry::IsModelCompatibleWithDomain(ModelId::MainstreamMaxwellEM, SimulationDomain::RadialAndScaleProfile), "MaxwellEM rejected for RadialAndScaleProfile", testName);

        // Verify epistemological provenance metadata on every registered model.
        for (const auto& desc : ModelRegistry::GetAllModels()) {
            ExpectTrue(desc.provenanceDetails != nullptr && desc.provenanceDetails[0] != '\0', "Model has non-empty provenanceDetails", testName);
        }
        ExpectTrue(ModelRegistry::FindById(ModelId::MainstreamNewtonian)->provenance == ModelProvenance::TextbookBaseline, "Newtonian is TextbookBaseline", testName);
        ExpectTrue(ModelRegistry::FindById(ModelId::ThinkerEssonne)->provenance == ModelProvenance::VerbatimEquation, "Essonne is VerbatimEquation", testName);
        ExpectTrue(ModelRegistry::FindById(ModelId::ThinkerPais)->provenance == ModelProvenance::HybridVerbatimAndAnsatz, "Pais is HybridVerbatimAndAnsatz", testName);
        ExpectTrue(ModelRegistry::FindById(ModelId::ThinkerVogelWillis)->provenance == ModelProvenance::HybridVerbatimAndAnsatz, "VogelWillis is HybridVerbatimAndAnsatz", testName);
        ExpectTrue(ModelRegistry::FindById(ModelId::ThinkerDanaan)->provenance == ModelProvenance::OperationalizedSpec, "Danaan is OperationalizedSpec", testName);
    }

    /**
     * @brief Verifies GPU FieldGrid2D Essonne spatial vacuum pressure & UnifiedSynthesis cavity.
     */
    void TestFieldGrid2DEssonneAndUnifiedGpu(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestFieldGrid2DEssonneAndUnifiedGpu";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        UniverseScene crystalScene{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "vogel_phryll_crystal.scenario.umgebung", crystalScene, err),
            "Load vogel_phryll_crystal.scenario.umgebung",
            testName
        );

        // 1. Test Thinker::Essonne 2D spatial vacuum field: interior pressure ratio in [0, 1] and depleted near center.
        auto sEssonne2D = ModelRegistry::CreateSolver(ModelId::ThinkerEssonne, SimulationDomain::FieldGrid2D, 0U);
        ExpectTrue(sEssonne2D != nullptr && sEssonne2D->Initialize(ctx, crystalScene), "Init Essonne FieldGrid2D", testName);
        ExpectTrue(sEssonne2D->StepMultiple(5ULL, 0.01), "Step Essonne FieldGrid2D", testName);

        FieldGrid2DSnapshot essSnap{};
        ExpectTrue(sEssonne2D->FetchFieldGrid2DSnapshot(essSnap), "Fetch Essonne 2D snapshot", testName);
        ExpectTrue(essSnap.width == 128U && essSnap.height == 128U, "128x128 grid dimensions", testName);
        ExpectTrue(essSnap.minVal < 0.05 && essSnap.maxVal > 0.7 && essSnap.maxVal <= 1.0,
                   "Essonne 2D vacuum pressure P/P0 depleted near critical core and approaches 1 outward", testName);
        sEssonne2D->Shutdown();

        // 2. Test Unified::Synthesis on FieldGrid2D cavity: finite, stable, non-zero field energy.
        auto sUni2D = ModelRegistry::CreateSolver(ModelId::UnifiedSynthesis, SimulationDomain::FieldGrid2D, 1U);
        ExpectTrue(sUni2D != nullptr && sUni2D->Initialize(ctx, crystalScene), "Init Unified::Synthesis FieldGrid2D", testName);
        ExpectTrue(sUni2D->StepMultiple(100ULL, 0.01), "Step Unified::Synthesis 100 FDTD steps", testName);

        SolverDiagnostics dUni{};
        ExpectTrue(sUni2D->ComputeSolverDiagnostics(dUni), "Unified 2D diagnostics", testName);
        ExpectTrue(std::isfinite(dUni.fieldEnergy) && dUni.fieldEnergy > 0.0, "Unified 2D cavity produces finite positive EM/Phryll energy", testName);
        ExpectTrue(dUni.effectiveCflNumber > 0.0 && dUni.effectiveCflNumber <= 0.451, "Unified 2D CFL strictly stable", testName);
        sUni2D->Shutdown();
    }

    /**
     * @brief Verifies RadialAndScaleProfile analytical curves for Haramein & Essonne vs SchwarzschildGR.
     */
    void TestRadialAndScaleProfileCurves(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestRadialAndScaleProfileCurves";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        // 1. Haramein Holographic Mass Sweep vs Mainstream::SchwarzschildGR across 42 orders of magnitude.
        UniverseScene harameinScene{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "haramein_holographic_scale.scenario.umgebung", harameinScene, err),
            "Load haramein_holographic_scale.scenario.umgebung",
            testName
        );

        auto sHara = ModelRegistry::CreateSolver(ModelId::ThinkerHaramein, SimulationDomain::RadialAndScaleProfile, 0U);
        auto sGR = ModelRegistry::CreateSolver(ModelId::MainstreamSchwarzschildGR, SimulationDomain::RadialAndScaleProfile, 1U);
        ExpectTrue(sHara != nullptr && sHara->Initialize(ctx, harameinScene), "Init Haramein RadialScaleProfile", testName);
        ExpectTrue(sGR != nullptr && sGR->Initialize(ctx, harameinScene), "Init SchwarzschildGR RadialScaleProfile", testName);

        RadialProfileSnapshot snapHara{}, snapGR{};
        ExpectTrue(sHara->FetchRadialProfileSnapshot(snapHara), "Fetch Haramein profile", testName);
        ExpectTrue(sGR->FetchRadialProfileSnapshot(snapGR), "Fetch GR profile", testName);
        ExpectTrue(snapHara.xValues.size() == 256U && snapGR.xValues.size() == 256U, "256 log-spaced scale samples", testName);

        // Verify Haramein m_h(r) agrees with classical Schwarzschild M_s(r) across all 42 orders of magnitude (< 1e-4 rel diff).
        double maxRelDiffMh = 0.0;
        for (std::size_t k = 0U; k < snapHara.xValues.size(); ++k) {
            const double rel = std::abs(snapHara.primaryYValues[k] - snapGR.primaryYValues[k]) / snapGR.primaryYValues[k];
            maxRelDiffMh = std::max(maxRelDiffMh, rel);
        }
        std::cout << "[RadialScale Haramein vs GR] maxRelDiff(m_h vs M_s)=" << maxRelDiffMh << std::endl;
        ExpectTrue(maxRelDiffMh < 1.0e-4, "Haramein m_h(r) matches GR Schwarzschild M_s(r) to < 1e-4 across 42 decades", testName);

        // 2. Chris Essonne Vacuum Horizon Profile vs Mainstream::SchwarzschildGR.
        UniverseScene essScene{};
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "essonne_vacuum_horizon.scenario.umgebung", essScene, err),
            "Load essonne_vacuum_horizon.scenario.umgebung",
            testName
        );

        auto sEss = ModelRegistry::CreateSolver(ModelId::ThinkerEssonne, SimulationDomain::RadialAndScaleProfile, 0U);
        auto sGREss = ModelRegistry::CreateSolver(ModelId::MainstreamSchwarzschildGR, SimulationDomain::RadialAndScaleProfile, 1U);
        ExpectTrue(sEss != nullptr && sEss->Initialize(ctx, essScene), "Init Essonne Horizon Profile", testName);
        ExpectTrue(sGREss != nullptr && sGREss->Initialize(ctx, essScene), "Init GR Horizon Profile", testName);

        RadialProfileSnapshot snapEss{}, snapGREss{};
        ExpectTrue(sEss->FetchRadialProfileSnapshot(snapEss), "Fetch Essonne horizon snapshot", testName);
        ExpectTrue(sGREss->FetchRadialProfileSnapshot(snapGREss), "Fetch GR horizon snapshot", testName);

        // At r = 0 (index 0) with R = 1.5 * Rs, Essonne Eq. (39) predicts P(0)/P_0 == 0 exactly!
        ExpectTrue(std::abs(snapEss.primaryYValues[0]) < 1.0e-14, "Essonne Eq. (39): P(0)/P_0 == 0 at R = 1.5 Rs", testName);

        // At r = 2.0 * Rs (find closest sample), Essonne c'(2Rs)/c == v_esc(2Rs)/c == 1/sqrt(2) (`8052.pdf` Eq. 43)!
        std::size_t idx2Rs = 0U;
        double bestDist = 1.0e9;
        for (std::size_t k = 0U; k < snapEss.xValues.size(); ++k) {
            const double d = std::abs(snapEss.xValues[k] - 2.0);
            if (d < bestDist) { bestDist = d; idx2Rs = k; }
        }
        const double cPrimeAt2Rs = snapEss.secondaryYValues[idx2Rs];
        const double vEscAt2Rs = snapGREss.secondaryYValues[idx2Rs];
        std::cout << "[RadialScale Essonne Horizon r~2Rs] c'(r)/c=" << cPrimeAt2Rs
                  << " vs v_esc(r)/c=" << vEscAt2Rs << std::endl;
        ExpectTrue(std::abs(cPrimeAt2Rs - vEscAt2Rs) < 0.02, "Essonne Eq. (43): c'(2Rs)/c intersects escape speed v_esc(2Rs)/c", testName);
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    TestDomainCompatibilityGuards();

    const auto caps = umgebung::core::SystemProfiler::ProfileSystem();
    const auto plan = umgebung::core::SystemProfiler::BuildAdaptationPlan(
        caps,
        umgebung::core::HardwareAdaptationConfig{.guiActive = false}
    );

    umgebung::compute::CUDAContext ctx;
    if (!ctx.Initialize(caps, plan)) {
        std::cerr << "[FATAL] Failed to initialize CUDAContext for SimulationDomainTests." << std::endl;
        return 1;
    }

    TestFieldGrid2DEssonneAndUnifiedGpu(ctx, scenesDir);
    TestRadialAndScaleProfileCurves(ctx, scenesDir);

    ctx.Shutdown();

    if (gFailedAssertions > 0) {
        std::cerr << "SimulationDomainTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All SimulationDomainTests passed." << std::endl;
    return 0;
}
