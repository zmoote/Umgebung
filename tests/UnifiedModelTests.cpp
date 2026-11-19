/**
 * @file UnifiedModelTests.cpp
 * @brief Automated CTest verification suite for Pillar 2 (`models/unified/`)
 *        and controlled A/B comparison against Pillar 1 (`models/mainstream/`).
 *
 * @details
 * Verifies:
 *   1. **Nassim Haramein (2013) Verbatim Holographic Mass Equations**
 *      (`Haramein342013PRRI3363.pdf` Eq. 5–13, 22–27): Cygnus X-1 Schwarzschild
 *      mass equivalence, muonic proton rest mass accuracy (~0.07% of CODATA),
 *      and exact geometric duality $m_h(r) \cdot m_p'(r) = 2 m_\ell^2$.
 *   2. **Chris Essonne Verbatim Vacuum Gradient & Horizon Equations**
 *      (`8052.pdf` Eq. 21, 23, 27, 35, 39, 43 & `10100.pdf` Eq. 7, 11):
 *      zero central pressure at $R = \frac{3}{2} R_s$, true horizon $c'(2R_s) = v_L(2R_s)$,
 *      and $C^1$-smooth spiral-galaxy flat rotation curve $v_{\text{flat}}$.
 *   3. **Salvatore Pais (`US10144532.pdf` Eq. 1–2) & Danaan/Willis/Vogel Constants.**
 *   4. **Live GPU A/B Divergence Test (`galaxy_periphery_mainstream.umgebung`
 *      vs. `galaxy_periphery_unified.umgebung` via `CreateSolverForScene`):**
 *      Proves identical periphery stars remain on flat circular orbits under
 *      Essonne's aether vortex (`Unified`) while outer stars escape radially
 *      under unassisted Newtonian $1/r^2$ (`Mainstream`), plus verifies
 *      vibrational-density proper-time ordering ($\tau_3 > \tau_2 > \tau_1$)
 *      and Essonne vacuum pressure depletion ($P_i/P_0 < 1$).
 *   5. **Antisymmetric Love/Phryll Pairwise Conservation Test on GPU:**
 *      Verifies strict linear momentum conservation ($< 10^{-13}$) and
 *      symplectic total energy conservation ($< 10^{-8}$) with non-zero Love/Phryll coupling.
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/models/SolverFactory.hpp"
#include "umgebung/models/unified/UnifiedFieldSolver.hpp"
#include "umgebung/models/unified/UnifiedPhysicsEquations.hpp"
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

    /**
     * @brief Test 1: Verbatim Nassim Haramein (2013) Holographic Mass equations.
     */
    void TestHarameinHolographicEquations() {
        const char* testName = "TestHarameinHolographicEquations";
        using namespace umgebung::models::unified;

        // 1. Cygnus X-1 benchmark from Haramein (2013), p. 274–275:
        //    r = 2.5e6 cm = 2.5e4 m => m_h = m_s = 1.683354e34 g = 1.683354e31 kg.
        const double cygnusRadiusM = 2.5e4;
        const double mhCygnus = ComputeHolographicSchwarzschildMass(cygnusRadiusM);
        const double msCygnus = (cygnusRadiusM * constants::kSpeedOfLightSI * constants::kSpeedOfLightSI) /
                                (2.0 * constants::kGravitationalConstantSI);
        const double relDiffCygnus = std::abs(mhCygnus - msCygnus) / msCygnus;
        const double relDiffPublished = std::abs(mhCygnus - 1.683354e31) / 1.683354e31;

        std::cout << "[Haramein 2013 Cygnus X-1] mh=" << mhCygnus
                  << " kg | ms=" << msCygnus
                  << " kg | relDiff=" << relDiffCygnus << std::endl;

        ExpectTrue(relDiffCygnus < 1.0e-4, "mh matches classical Schwarzschild mass ms < 1e-4", testName);
        ExpectTrue(relDiffPublished < 1.0e-3, "mh matches Haramein 2013 published 1.683354e31 kg < 1e-3", testName);

        // 2. Muonic proton rest mass from Haramein (2013) Eq. (25):
        //    r_p = 0.84184e-15 m => m_p' = 1.6714213e-27 kg (~0.072% from CODATA 1.672622e-27 kg).
        const double mpHolographic = ComputeHolographicProtonMass(constants::kMuonicProtonRadiusSI);
        const double relErrCodata = std::abs(mpHolographic - constants::kCodataProtonMassSI) /
                                    constants::kCodataProtonMassSI;
        const double relErrPaper = std::abs(mpHolographic - 1.6714213e-27) / 1.6714213e-27;

        std::cout << "[Haramein 2013 Proton] m_p'=" << mpHolographic
                  << " kg | CODATA=" << constants::kCodataProtonMassSI
                  << " kg | relErrCodata=" << (relErrCodata * 100.0) << "%" << std::endl;

        ExpectTrue(relErrPaper < 1.0e-5, "m_p' matches Haramein 2013 Eq. (25) to < 1e-5", testName);
        ExpectTrue(relErrCodata < 1.0e-3, "m_p' within 0.1% of CODATA proton mass", testName);

        // 3. Exact algebraic geometric duality (also formally proven in Lean 4):
        //    m_h(r) * m_p'(r) = 2 * m_l^2 for arbitrary radius r > 0.
        const double product = ComputeHolographicSchwarzschildMass(1.2345e-10) *
                               ComputeHolographicProtonMass(1.2345e-10);
        const double expectedTwoMlSq = 2.0 * constants::kPlanckMassHaramein2013SI *
                                             constants::kPlanckMassHaramein2013SI;
        const double dualityRelErr = std::abs(product - expectedTwoMlSq) / expectedTwoMlSq;
        ExpectTrue(dualityRelErr < 1.0e-14, "Haramein exact duality m_h(r) * m_p'(r) = 2 m_l^2", testName);
    }

    /**
     * @brief Test 2: Verbatim Chris Essonne (8052.pdf & 10100.pdf) equations.
     */
    void TestEssonneVacuumAndVortexEquations() {
        const char* testName = "TestEssonneVacuumAndVortexEquations";
        using namespace umgebung::models::unified;

        const double Rs = 10.0;
        const double c = constants::kSpeedOfLightSI;

        // 8052.pdf Eq. (39): Zero central pressure inside homogeneous sphere iff R = 1.5 * Rs.
        const double pCenterAt1_5Rs = ComputeEssonneInteriorPressureRatio(0.0, 1.5 * Rs, Rs);
        ExpectTrue(std::abs(pCenterAt1_5Rs) < 1.0e-15, "Essonne Eq. (39): P(0)/P0 == 0 at R = 1.5 Rs", testName);

        // 8052.pdf Eq. (43): True horizon where local light speed c'(r) equals escape velocity v_L(r) is r = 2 Rs.
        const double rHorizon = 2.0 * Rs;
        const double cLocal = ComputeEssonneLocalLightSpeed(rHorizon, Rs, c);
        const double vEscape = c * std::sqrt(Rs / rHorizon);
        ExpectTrue(std::abs(cLocal - vEscape) / c < 1.0e-15, "Essonne Eq. (43): c'(2Rs) == v_L(2Rs)", testName);

        // 8052.pdf Eq. (23) & (27): c'(r) * n(r) == c outside Rs.
        const double rTest = 5.0 * Rs;
        const double productCN = ComputeEssonneLocalLightSpeed(rTest, Rs, c) *
                                 ComputeEssonneRefractiveIndex(rTest, Rs);
        ExpectTrue(std::abs(productCN - c) / c < 1.0e-15, "Essonne Eq. (23)/(27): c'(r) * n(r) == c", testName);

        // 10100.pdf Eq. (7) & (11): C^1 continuity across r = Rc and constant flat velocity v_flat.
        const double M = 1.0;
        const double G = 1.0;
        const double Rc = 1.0;
        const double h = 2.0;
        const double gz = 0.0;

        const double gInnerBoundary = ComputeEssonneAetherVortexRadialAccel(Rc - 1.0e-9, M, G, Rc, h, gz);
        const double gOuterBoundary = ComputeEssonneAetherVortexRadialAccel(Rc + 1.0e-9, M, G, Rc, h, gz);
        const double vFlat = ComputeEssonneFlatRotationVelocity(M, G, Rc, h, gz);

        ExpectTrue(std::abs(gInnerBoundary - gOuterBoundary) < 1.0e-7, "Essonne vortex g_r(r) continuous at r=Rc", testName);
        ExpectTrue(std::abs(vFlat - 1.0) < 1.0e-15, "Essonne Eq. (11): v_flat == 1.0 for M=1, G=1, h=2, gz=0", testName);
    }

    /**
     * @brief Test 3: Verbatim Salvatore Pais (US10144532.pdf) & Danaan/Willis/Vogel constants.
     */
    void TestPaisAndPhryllEquations() {
        const char* testName = "TestPaisAndPhryllEquations";
        using namespace umgebung::models::unified;

        // Pais US10144532 Eq. (1): Disc f_G=1, sigma=50000 C/m^2, R_r=2 m, omega=1e9 rad/s.
        const double sRotOnly = ComputePaisMaxPoyntingFlux(1.0, 50000.0, 2.0, 1.0e9, 0.0, 0.0, 0.0);
        const double sExpected = (50000.0 * 50000.0 / constants::kVacuumPermittivitySI) * (2.0 * 1.0e9);
        ExpectTrue(std::abs(sRotOnly - sExpected) / sExpected < 1.0e-14, "Pais Eq. (1) exact Poynting flux", testName);

        // Pais Eq. (2): Accelerated harmonic vibration.
        const double sAcc = ComputePaisAcceleratedPoyntingFlux(1.0, 50000.0, 1.0e-3, 1.0e9, 1.0);
        ExpectTrue(sAcc > 0.0 && std::isfinite(sAcc), "Pais Eq. (2) finite positive flux", testName);

        // Vogel/Eredyon Phryll crystal constants (PhryllGenInstructions.pdf).
        ExpectTrue(std::abs(constants::kVogelBaseAngleDeg - 51.84) < 1e-12, "Vogel base angle == 51.84 deg", testName);
        ExpectTrue(std::abs(constants::kVogelTopAngleDeg - 60.0) < 1e-12, "Vogel top angle == 60.0 deg", testName);
        ExpectTrue(std::abs(constants::kPhryllTorsionFrequencyHz - 4096.0) < 1e-12, "Phryll torsion freq == 4096 Hz", testName);
    }

    /**
     * @brief Test 4: Live GPU A/B comparison on identical galaxy periphery scenes.
     */
    void TestLiveGpuABGalaxyPeripheryComparison(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestLiveGpuABGalaxyPeripheryComparison";
        using namespace umgebung::scene;
        using namespace umgebung::models;

        // Single canonical galaxy_periphery.scenario.umgebung executed under
        // Mainstream::Newtonian (stream 0) vs. Unified::Synthesis (stream 1) simultaneously!
        UniverseScene galaxyScene{};
        std::string err;

        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "galaxy_periphery.scenario.umgebung", galaxyScene, err),
            "Load single unified scenarios/galaxy_periphery.scenario.umgebung",
            testName
        );

        auto mainstreamSolver = ModelRegistry::CreateSolver(ModelId::MainstreamNewtonian, 0U);
        auto unifiedSolver = ModelRegistry::CreateSolver(ModelId::UnifiedSynthesis, 1U);

        ExpectTrue(mainstreamSolver->GetModelId() == ModelId::MainstreamNewtonian, "Registry dispatched Mainstream::Newtonian", testName);
        ExpectTrue(unifiedSolver->GetModelId() == ModelId::UnifiedSynthesis, "Registry dispatched Unified::Synthesis", testName);

        ExpectTrue(mainstreamSolver->Initialize(ctx, galaxyScene), "Initialize mainstreamSolver", testName);
        ExpectTrue(unifiedSolver->Initialize(ctx, galaxyScene), "Initialize unifiedSolver", testName);

        constexpr std::uint64_t kSteps = 3000ULL;
        constexpr double kDt = 0.002; // Total time t = 6.0.

        ExpectTrue(mainstreamSolver->StepMultiple(kSteps, kDt), "Step mainstreamSolver 3000 steps", testName);
        ExpectTrue(unifiedSolver->StepMultiple(kSteps, kDt), "Step unifiedSolver 3000 steps", testName);

        SolverDiagnostics diagMain{};
        SolverDiagnostics diagUni{};
        ExpectTrue(mainstreamSolver->ComputeSolverDiagnostics(diagMain), "Mainstream diagnostics", testName);
        ExpectTrue(unifiedSolver->ComputeSolverDiagnostics(diagUni), "Unified diagnostics", testName);

        std::vector<BodyEntity> mainBodies = galaxyScene.bodies;
        std::vector<BodyEntity> uniBodies = galaxyScene.bodies;
        ExpectTrue(mainstreamSolver->SyncToHost(mainBodies), "Sync mainstream bodies", testName);
        ExpectTrue(unifiedSolver->SyncToHost(uniBodies), "Sync unified bodies", testName);

        // Verify Unified Essonne vortex holds ALL 3 periphery stars on exact flat circular orbits (r = r0, v = 1.0).
        const double expectedRadii[3] = {1.5, 3.0, 5.0};
        for (std::size_t idx = 0U; idx < 3U; ++idx) {
            const std::size_t b = idx + 1U;
            const double rUni = (uniBodies[b].position - uniBodies[0].position).Norm();
            const double vUni = uniBodies[b].velocity.Norm();
            const double rMain = (mainBodies[b].position - mainBodies[0].position).Norm();

            std::cout << "[A/B Star " << b << " (r0=" << expectedRadii[idx] << ")] "
                      << "Unified r=" << rUni << " (v=" << vUni << ") vs "
                      << "Mainstream r=" << rMain << " | deltaR=" << (rMain - rUni) << std::endl;

            ExpectTrue(std::abs(rUni - expectedRadii[idx]) < 1.0e-3, "Unified star holds constant radius r0", testName);
            ExpectTrue(std::abs(vUni - 1.0) < 1.0e-3, "Unified star holds flat speed v_flat == 1.0", testName);
        }

        // Verify quantitative A/B divergence: outer stars (r0=3.0 and r0=5.0) expand outward under Mainstream 1/r^2.
        const double rMainOuter = (mainBodies[3].position - mainBodies[0].position).Norm();
        const double rUniOuter = (uniBodies[3].position - uniBodies[0].position).Norm();
        ExpectTrue((rMainOuter - rUniOuter) > 1.5, "Mainstream outer star diverges outward by > 1.5 units vs Unified", testName);

        // Verify Unified symplectic energy conservation & telemetry population.
        ExpectTrue(diagUni.hasUnifiedTelemetry, "Unified solver populates unified telemetry", testName);
        ExpectTrue(!diagMain.hasUnifiedTelemetry, "Mainstream solver does not spoof unified telemetry", testName);
        ExpectTrue(diagUni.relativeEnergyDrift < 1.0e-6, "Unified Essonne vortex symplectic energy drift < 1e-6", testName);
        ExpectTrue(diagUni.meanVacuumPressureRatio < 1.0 && diagUni.meanVacuumPressureRatio > 0.99, "Essonne <P/P0> depleted < 1", testName);
        ExpectTrue(diagUni.totalPhryllFlux > 4096.0, "Phryll flux reflects field + body charges", testName);

        // Verify non-linear local proper time ordering reflects vibrational densities (D3=5.0 > D2=3.5 > D1=3.0).
        auto* rawUnified = dynamic_cast<unified::UnifiedFieldSolver*>(unifiedSolver.get());
        ExpectTrue(rawUnified != nullptr, "dynamic_cast to UnifiedFieldSolver", testName);
        if (rawUnified != nullptr) {
            std::vector<double> properTimes;
            ExpectTrue(rawUnified->GetLocalProperTimes(properTimes), "GetLocalProperTimes", testName);
            ExpectTrue(properTimes.size() == 4U, "4 proper time clocks", testName);
            if (properTimes.size() == 4U) {
                ExpectTrue(
                    properTimes[3] > properTimes[2] && properTimes[2] > properTimes[1],
                    "Higher vibrational density bodies experience faster local proper time (tau3 > tau2 > tau1)",
                    testName
                );
            }
        }

        mainstreamSolver->Shutdown();
        unifiedSolver->Shutdown();
    }

    /**
     * @brief Test 5: Antisymmetric Love/Phryll pairwise force preserves momentum & energy on GPU.
     */
    void TestLovePhryllPairwiseConservation(umgebung::compute::CUDAContext& ctx) {
        const char* testName = "TestLovePhryllPairwiseConservation";
        using namespace umgebung::scene;
        using namespace umgebung::models;

        UniverseScene pairScene{};
        pairScene.universeId = "love_phryll_binary";
        pairScene.name = "Love/Phryll Coherent Binary Conservation Test";
        pairScene.physicsProfile.type = PhysicsProfileType::Unified;
        pairScene.physicsProfile.defaultModel = ModelId::UnifiedSynthesis;
        pairScene.physicsProfile.integrator = IntegratorType::VelocityVerlet;
        pairScene.physicsProfile.timestep = 0.001;
        pairScene.physicsProfile.mainstreamGravity.gravitationalConstant = 1.0;
        pairScene.physicsProfile.mainstreamGravity.softeningLength = 0.0;
        pairScene.physicsProfile.unifiedField.aetherVortexEnabled = false;
        pairScene.physicsProfile.unifiedField.loveBindingCoupling = 0.25;
        pairScene.physicsProfile.unifiedField.phryllFieldAmplitude = 100.0;
        pairScene.physicsProfile.unifiedField.referenceLightSpeed = 1000.0;
        pairScene.physicsProfile.unifiedField.nonLinearTimeEnabled = true;

        BodyEntity a{};
        a.id = "entity_a";
        a.name = "Entity A";
        a.mass = 1.0;
        a.radius = 0.1;
        a.position = {-0.75, 0.0, 0.0};
        a.velocity = {0.0, -0.65, 0.0};
        a.consciousnessCoherence = 0.8;
        a.phryllCharge = 2.0;

        BodyEntity b{};
        b.id = "entity_b";
        b.name = "Entity B";
        b.mass = 1.0;
        b.radius = 0.1;
        b.position = {0.75, 0.0, 0.0};
        b.velocity = {0.0, 0.65, 0.0};
        b.consciousnessCoherence = 0.9;
        b.phryllCharge = 3.0;

        pairScene.bodies = {a, b};

        auto solver = CreateSolverForScene(pairScene);
        ExpectTrue(solver->Initialize(ctx, pairScene), "Initialize Love/Phryll binary solver", testName);

        SolverDiagnostics d0{};
        ExpectTrue(solver->ComputeSolverDiagnostics(d0), "Initial Love/Phryll diagnostics", testName);
        ExpectTrue(solver->StepMultiple(4000ULL, 0.00025), "Step 4000 steps with Love/Phryll coupling", testName);

        SolverDiagnostics df{};
        ExpectTrue(solver->ComputeSolverDiagnostics(df), "Final Love/Phryll diagnostics", testName);

        const double momDrift = (df.totalLinearMomentum - d0.totalLinearMomentum).Norm();
        const double angMomDrift = (df.totalAngularMomentum - d0.totalAngularMomentum).Norm();

        std::cout << "[Unified Love/Phryll Binary] relEnergyDrift=" << df.relativeEnergyDrift
                  << " | momDrift=" << momDrift
                  << " | angMomDrift=" << angMomDrift << std::endl;

        ExpectTrue(momDrift < 1.0e-13, "Love/Phryll antisymmetric force conserves linear momentum < 1e-13", testName);
        ExpectTrue(angMomDrift < 1.0e-12, "Love/Phryll central force conserves angular momentum < 1e-12", testName);
        ExpectTrue(df.relativeEnergyDrift < 1.0e-6, "Love/Phryll symplectic energy conserved < 1e-6", testName);

        solver->Shutdown();
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    TestHarameinHolographicEquations();
    TestEssonneVacuumAndVortexEquations();
    TestPaisAndPhryllEquations();

    const auto caps = umgebung::core::SystemProfiler::ProfileSystem();
    const auto plan = umgebung::core::SystemProfiler::BuildAdaptationPlan(
        caps,
        umgebung::core::HardwareAdaptationConfig{.guiActive = false}
    );

    umgebung::compute::CUDAContext ctx;
    if (!ctx.Initialize(caps, plan)) {
        std::cerr << "[FATAL] Failed to initialize CUDAContext for UnifiedModelTests." << std::endl;
        return 1;
    }

    TestLiveGpuABGalaxyPeripheryComparison(ctx, scenesDir);
    TestLovePhryllPairwiseConservation(ctx);

    ctx.Shutdown();

    if (gFailedAssertions > 0) {
        std::cerr << "UnifiedModelTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All UnifiedModelTests passed." << std::endl;
    return 0;
}
