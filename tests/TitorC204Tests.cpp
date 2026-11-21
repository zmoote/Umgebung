/**
 * @file TitorC204Tests.cpp
 * @brief Automated CTest verification suite for the **GE Army Model C204 Gravity
 *        Distortion Time Displacement Unit** (`Thinker::TitorC204`):
 *          1. Verbatim `C204_Operators_Manual.pdf` (`TMIM-200S/4005`) tables
 *             (50/75/100% power, Coulombs, penetration rates, $v_t$, VGL).
 *          2. Foundational General Relativity invariants (van Stockum/Tipler
 *             $FL + M^2 = r^2$, CTL onset, supercritical $aR > 0.5$ CTCs,
 *             Kerr-Newman horizons & photon orbits, Van Den Broeck screening,
 *             $170^\circ$ aft X-ray venting cone, Borda 3-computer voting).
 *          3. FP64 CUDA `TitorC204Solver` end-to-end operational lifecycle
 *             (`VglBaselineLock -> RampUp [UV Burst] -> TemporalTransit [Blackout]
 *             -> VglPulseCheck [240 s] -> ArrivalCooldown`), concentric Negative /
 *             Null / Positive Time Event Horizons, calibrated `1.5–2.0 G` cabin
 *             acceleration, `100–120 F` hull temperature, and Automatic Origin
 *             Return failsafes (VGL gravity anomaly & `> 2500 kg` mass overload).
 *          4. Simultaneous GPU A/B comparison (`Thinker::TitorC204` vs
 *             `Thinker::Essonne` on `FieldGrid2D` and `Thinker::TitorC204` vs
 *             `Mainstream::SchwarzschildGR` on `RadialAndScaleProfile`).
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/models/thinkers/TitorC204Model.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

    int gFailedAssertions = 0;

    /**
     * @brief Records a failed verification assertion without aborting remaining checks.
     */
    void ExpectTrue(bool condition, const char* exprText, const char* testName) {
        if (!condition) {
            std::cerr << "[FAIL] " << testName << ": assertion failed -> " << exprText << std::endl;
            ++gFailedAssertions;
        }
    }

    /**
     * @brief Verifies verbatim C204 Operator's Manual calibration tables and
     *        exact foundational relativity identities (matching `TitorC204.lean`).
     */
    void TestVerbatimManualAndRelativityMath() {
        const char* testName = "TestVerbatimManualAndRelativityMath";
        using namespace umgebung::models::thinkers;

        // 1. Verbatim manual electrical charge & wave penetration rate tables.
        ExpectTrue(std::abs(ComputeC204ChargeForPower(50.0) - 126000.0) < 1e-9, "50% power == 126,000 C", testName);
        ExpectTrue(std::abs(ComputeC204ChargeForPower(75.0) - 201600.0) < 1e-9, "75% power == 201,600 C", testName);
        ExpectTrue(std::abs(ComputeC204ChargeForPower(100.0) - 288000.0) < 1e-9, "100% power == 288,000 C", testName);
        ExpectTrue(ComputeC204DirectionalCharge(100.0, true) < ComputeC204DirectionalCharge(100.0, false),
                   "Forward time travel requires strictly less electrical charge than backward CTC travel", testName);
        ExpectTrue(std::abs(ComputeC204DirectionalCharge(100.0, true) - 216000.0) < 1e-9,
                   "100% power forward travel charge == 216,000 C (-25%)", testName);

        ExpectTrue(std::abs(ComputeC204PenetrationRateYrsPerHr(50.0) - 3.4) < 1e-12, "50% rate == 3.4 yrs/hr", testName);
        ExpectTrue(std::abs(ComputeC204PenetrationRateYrsPerHr(75.0) - 6.2) < 1e-12, "75% rate == 6.2 yrs/hr", testName);
        ExpectTrue(std::abs(ComputeC204PenetrationRateYrsPerHr(100.0) - 10.0) < 1e-12, "100% rate == 10.0 yrs/hr", testName);

        const double vt100 = ComputeC204CoordinateTimeVelocity(10.0);
        ExpectTrue(std::abs(vt100 - 87660.0) < 1e-6, "100% coordinate time velocity == 87,660 s/s", testName);
        ExpectTrue(std::abs(ComputeC204PowerLawTimeVelocity(288000.0) - 87660.0) < 1e-6, "Power-law v_t(288k C) == 87,660 s/s", testName);

        // 2. Van Stockum / Tipler (1974) interior determinant FL + M^2 = r^2 & CTL onset.
        const double aCyl = 0.8;
        const double rCyl = 0.9;
        const double L = ComputeVanStockumInteriorL(rCyl, aCyl);
        const double M = aCyl * rCyl * rCyl;
        ExpectTrue(std::abs((1.0 * L + M * M) - (rCyl * rCyl)) < 1e-14, "van Stockum FL + M^2 == r^2", testName);
        ExpectTrue(std::abs(ComputeVanStockumInteriorL(1.0 / aCyl, aCyl)) < 1e-14, "CTL boundary L(1/a) == 0", testName);
        ExpectTrue(ComputeVanStockumInteriorL(1.5 / aCyl, aCyl) < 0.0, "Interior L(r) < 0 for r > 1/a (CTLs)", testName);

        // Supercritical exterior (aR = 0.75 > 0.5) oscillates negative for large r/R.
        bool foundNegativeExteriorL = false;
        for (int k = 1; k <= 50; ++k) {
            const double rProbe = 1.0 + 0.5 * static_cast<double>(k);
            if (ComputeTiplerSupercriticalExteriorL(rProbe, 1.0, 0.75) < 0.0) {
                foundNegativeExteriorL = true;
                break;
            }
        }
        ExpectTrue(foundNegativeExteriorL, "Supercritical Tipler exterior (aR=0.75 > 0.5) develops L(r) < 0 CTC zone", testName);

        // 3. Kerr-Newman horizon discriminant, super-extremality, and equatorial photon orbits.
        ExpectTrue(std::abs(ComputeKerrNewmanDelta(2.0, 1.0, 0.0, 0.0)) < 1e-15, "Schwarzschild limit Delta(2M) == 0", testName);
        ExpectTrue(std::abs(ComputeKerrNewmanDelta(1.0, 1.0, 0.6, 0.8)) < 1e-15, "Extremal double root Delta(M) == 0 for a^2+Q^2=M^2", testName);
        ExpectTrue(IsKerrNewmanSuperExtremal(1.0, 0.92, 0.45), "C204 (a/M=0.92, Q/M=0.45) is super-extremal naked singularity", testName);

        const double r1Zero = ComputeKerrProgradePhotonRadius(1.0, 0.0);
        const double r2Zero = ComputeKerrRetrogradePhotonRadius(1.0, 0.0);
        ExpectTrue(std::abs(r1Zero - 3.0) < 1e-12 && std::abs(r2Zero - 3.0) < 1e-12, "Kerr photon orbits collapse to 3M at a=0", testName);

        const double r1Spin = ComputeKerrProgradePhotonRadius(1.0, 0.92);
        const double r2Spin = ComputeKerrRetrogradePhotonRadius(1.0, 0.92);
        ExpectTrue(r1Spin >= 1.0 && r1Spin < 3.0 && r2Spin > 3.0 && r2Spin <= 4.0, "Kerr photon orbits satisfy M <= r1 < 3M < r2 <= 4M", testName);

        // 4. Van Den Broeck ellipsoidal shaping & 170-deg aft X-ray venting cone.
        ExpectTrue(std::abs(ComputeVanDenBroeckShapingFactor(0.0, 4.0) - 1.0) < 1e-14, "Van Den Broeck f(0) == 1.0", testName);
        ExpectTrue(ComputeVanDenBroeckShapingFactor(2.0, 4.0) < 1.0e-3, "Van Den Broeck exterior screened (f(2) < 1e-3)", testName);
        ExpectTrue(IsPointInC204XRayVentCone(-5.0, 0.0, 0.0, 170.0), "Aft point (-5, 0) inside 170 deg X-ray cone", testName);
        ExpectTrue(!IsPointInC204XRayVentCone(+1.8, 0.0, 0.0, 170.0), "Forward pilot cabin (+1.8, 0) safely outside aft X-ray cone", testName);

        // 5. Triple-computer Borda error-correcting consensus voting.
        double consensus = 0.0;
        double spread = 0.0;
        ExpectTrue(EvaluateC204BordaConsensus(1.0, 1.0, 1.0, true, true, true, consensus, spread), "Borda 3-computer unanimous succeeds", testName);
        ExpectTrue(std::abs(consensus - 1.0) < 1e-15 && spread < 1e-15, "Borda 3-computer unanimous == 1.0", testName);

        // Single faulty sensor glitch (+5.0) is rejected cleanly by Borda consensus!
        ExpectTrue(EvaluateC204BordaConsensus(1.0, 1.001, 6.0, true, true, true, consensus, spread), "Borda rejects single outlier channel", testName);
        ExpectTrue(std::abs(consensus - 1.0005) < 1e-9 && spread < 0.002, "Borda outlier channel excluded from consensus", testName);

        // Degraded 2-computer mode (1 disabled per manual) still operates cleanly.
        ExpectTrue(EvaluateC204BordaConsensus(1.0, 1.0, 99.0, true, true, false, consensus, spread), "Borda operates with 1 computer disabled", testName);
        ExpectTrue(std::abs(consensus - 1.0) < 1e-15, "2-computer degraded consensus == 1.0", testName);
        ExpectTrue(!EvaluateC204BordaConsensus(1.0, 1.0, 1.0, true, false, false, consensus, spread), "Fails safely with < 2 active computers", testName);
    }

    /**
     * @brief Verifies the FP64 CUDA `TitorC204Solver` mission lifecycle, concentric
     *        metric horizons, 1.5-2.0 G cabin gradient, hull temp, VGL checks,
     *        Automatic Origin Return failsafes, and simultaneous A/B comparisons.
     */
    void TestC204GpuLifecycleAndFailsafes(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestC204GpuLifecycleAndFailsafes";
        using namespace umgebung::models;
        using namespace umgebung::scene;

        UniverseScene c204Scene{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "titor_c204_unit.scenario.umgebung", c204Scene, err),
            "Load titor_c204_unit.scenario.umgebung",
            testName
        );
        ExpectTrue(c204Scene.domain == SimulationDomain::FieldGrid2D, "C204 unit scenario uses FieldGrid2D", testName);
        ExpectTrue(c204Scene.physicsProfile.defaultModel == ModelId::ThinkerTitorC204, "Default model is Thinker::TitorC204", testName);

        // 1. Run primary mission through VGL Lock -> RampUp (Rainbow/UV) -> Transit (Blackout) -> VGL Pulse!
        auto solver = ModelRegistry::CreateSolver(ModelId::ThinkerTitorC204, SimulationDomain::FieldGrid2D, 0U);
        ExpectTrue(solver != nullptr && solver->Initialize(ctx, c204Scene), "Initialize TitorC204Solver on GPU", testName);

        // Step into final second of Ramp-Up (t = 11.5 s: 2s VGL lock + 9.5s of 10s ramp-up -> UV burst active!).
        ExpectTrue(solver->StepMultiple(23ULL, 0.5), "Step into UV burst ramp-up window", testName);
        SolverDiagnostics dUv{};
        ExpectTrue(solver->ComputeSolverDiagnostics(dUv), "Compute diagnostics during UV burst", testName);
        ExpectTrue(dUv.hasC204Telemetry, "Diagnostics report hasC204Telemetry == true", testName);
        ExpectTrue(dUv.c204UvBurstActive, "1-second UV burst flag active during final second of ramp-up", testName);
        ExpectTrue(!dUv.c204OpticalBlackoutActive, "Optical blackout not yet closed prior to full transit", testName);

        // Step past ramp-up and across 240s VGL check into deep TemporalTransit (t = 300 s).
        ExpectTrue(solver->StepMultiple(288ULL, 1.0), "Step 288 s through blackout transit & first 240 s VGL pulse", testName);
        SolverDiagnostics dTransit{};
        ExpectTrue(solver->ComputeSolverDiagnostics(dTransit), "Compute diagnostics during transit", testName);

        std::cout << "[TitorC204 GPU Transit] state=" << dTransit.c204OperationalState
                  << " | year=" << dTransit.c204CurrentWorldlineYear
                  << " | cabinG=" << dTransit.c204CabinGForce
                  << " | tempF=" << dTransit.c204ExteriorTempF
                  << " | div=" << dTransit.c204WorldlineDivergencePct << "%"
                  << " | VGL=" << dTransit.c204VglCheckCount
                  << " | negCells=" << dTransit.c204NegativeHorizonCells
                  << " | nullCells=" << dTransit.c204NullHorizonCells << std::endl;

        ExpectTrue(dTransit.c204OpticalBlackoutActive, "Total optical blackout active during transit", testName);
        ExpectTrue(!dTransit.c204OriginReturnTriggered, "No abort on nominal mission", testName);
        ExpectTrue(dTransit.c204VglCheckCount >= 1ULL, "At least 1 VGL 240s check executed", testName);
        ExpectTrue(dTransit.c204ElapsedDisplacedYears < -0.5, "Worldline displacing backward from 2036 toward 1975", testName);
        ExpectTrue(dTransit.c204CurrentWorldlineYear < 2035.5, "Current calendar year decreased below 2035.5", testName);
        ExpectTrue(dTransit.c204CabinGForce >= 1.5 && dTransit.c204CabinGForce <= 2.0,
                   "Cabin G-force strictly within verbatim 1.5-2.0 G range at 100% power", testName);
        ExpectTrue(dTransit.c204ExteriorTempF >= 100.0 && dTransit.c204ExteriorTempF <= 120.0,
                   "Exterior hull temperature strictly within verbatim 100-120 F range", testName);
        ExpectTrue(dTransit.c204MetricGradG00AtCabin > 2.5e-16 && dTransit.c204MetricGradG00AtCabin < 5.0e-16,
                   "Cabin metric gradient |grad g_00| = 2a/c^2 matches Turn 9 constraint (~3.3-4.3e-16 m^-1)", testName);
        ExpectTrue(dTransit.c204NegativeHorizonCells > 0ULL &&
                   dTransit.c204NullHorizonCells > 0ULL &&
                   dTransit.c204PositiveHorizonCells > 0ULL,
                   "Backward CTC GPU 2D metric grid exhibits Negative, Null, and Positive Time Horizons", testName);
        ExpectTrue(!dTransit.c204IsForwardTravel, "Backward mission reports c204IsForwardTravel == false", testName);

        // 1b. Verify Forward-in-Time mission (+2.0 yrs): less energy, ZERO Negative CTC cells,
        //     active Positive Time Field cells > 0, and Null Time cancellation shell > 0!
        UniverseScene forwardScene = c204Scene;
        forwardScene.physicsProfile.unifiedField.titorC204.targetDisplacementYears = +2.0;
        auto fwdSolver = ModelRegistry::CreateSolver(ModelId::ThinkerTitorC204, SimulationDomain::FieldGrid2D, 1U);
        ExpectTrue(fwdSolver != nullptr && fwdSolver->Initialize(ctx, forwardScene), "Init Forward (+2.0 yr) TitorC204Solver", testName);
        ExpectTrue(fwdSolver->StepMultiple(300ULL, 1.0), "Step Forward (+2.0 yr) solver 300 s", testName);
        SolverDiagnostics dFwd{};
        ExpectTrue(fwdSolver->ComputeSolverDiagnostics(dFwd), "Forward (+2.0 yr) diagnostics", testName);
        std::cout << "[TitorC204 GPU Forward +2yr] year=" << dFwd.c204CurrentWorldlineYear
                  << " | Q=" << dFwd.c204InjectedChargeCoulombs << " C (used=" << dFwd.c204ExpendedChargeCoulombs << " C)"
                  << " | cabinG=" << dFwd.c204CabinGForce
                  << " | Neg=" << dFwd.c204NegativeHorizonCells
                  << " | Null=" << dFwd.c204NullHorizonCells
                  << " | Pos=" << dFwd.c204PositiveHorizonCells << std::endl;
        ExpectTrue(dFwd.c204IsForwardTravel, "Forward mission reports c204IsForwardTravel == true", testName);
        ExpectTrue(dFwd.c204CurrentWorldlineYear > 2036.5, "Forward mission advances calendar year into future", testName);
        ExpectTrue(dFwd.c204InjectedChargeCoulombs < dTransit.c204InjectedChargeCoulombs,
                   "Forward mission uses strictly lower active Coulomb charge than backward mission", testName);
        ExpectTrue(dFwd.c204ExpendedChargeCoulombs < dTransit.c204ExpendedChargeCoulombs,
                   "Forward mission expends strictly less total charge than backward mission", testName);
        ExpectTrue(dFwd.c204NegativeHorizonCells == 0ULL,
                   "Forward (+2.0 yr) mode produces ZERO Negative Time CTC cells", testName);
        ExpectTrue(dFwd.c204NullHorizonCells > 0ULL && dFwd.c204PositiveHorizonCells > 0ULL,
                   "Forward (+2.0 yr) mode exhibits active Null cancellation shell and Positive Time Field cells", testName);
        ExpectTrue(dFwd.c204CabinGForce >= 1.5 && dFwd.c204CabinGForce <= 2.0,
                   "Forward cabin G-force remains within 1.5-2.0 G window", testName);
        fwdSolver->Shutdown();

        // Verify outer corner of 30m x 30m grid is screened flat Minkowski (|1 + g_00| < 1e-3)!
        FieldGrid2DSnapshot snap{};
        ExpectTrue(solver->FetchFieldGrid2DSnapshot(snap), "Fetch C204 2D GPU snapshot", testName);
        ExpectTrue(snap.width == 128U && snap.height == 128U, "128x128 C204 GPU grid", testName);
        ExpectTrue(snap.scalarField.front() < 1.0e-3 && snap.scalarField.back() < 1.0e-3,
                   "Van Den Broeck exterior corners are flat Minkowski (|1 + g_00| < 1e-3)", testName);

        // 2. Test Automatic Origin Return on VGL Gravity Anomaly Injection!
        UniverseScene vglFaultScene = c204Scene;
        vglFaultScene.physicsProfile.unifiedField.titorC204.simulatedVglAnomaly = 0.25; // Exceeds 0.15 threshold
        auto faultSolver = ModelRegistry::CreateSolver(ModelId::ThinkerTitorC204, SimulationDomain::FieldGrid2D, 1U);
        ExpectTrue(faultSolver != nullptr && faultSolver->Initialize(ctx, vglFaultScene), "Init VGL fault test solver", testName);
        ExpectTrue(faultSolver->StepMultiple(5ULL, 1.0), "Step VGL fault test solver", testName);
        SolverDiagnostics dFault{};
        ExpectTrue(faultSolver->ComputeSolverDiagnostics(dFault), "Diagnostics on VGL fault", testName);
        ExpectTrue(dFault.c204OriginReturnTriggered, "VGL gravity anomaly triggers Automatic Origin Return", testName);
        faultSolver->Shutdown();

        // 3. Test Automatic Origin Return on Pre-Flight Payload Overload (> 2500 kg)!
        UniverseScene overloadScene = c204Scene;
        overloadScene.physicsProfile.unifiedField.titorC204.payloadMassKg = 2750.0;
        auto overloadSolver = ModelRegistry::CreateSolver(ModelId::ThinkerTitorC204, SimulationDomain::FieldGrid2D, 1U);
        ExpectTrue(overloadSolver != nullptr && overloadSolver->Initialize(ctx, overloadScene), "Init payload overload solver", testName);
        SolverDiagnostics dOverload{};
        ExpectTrue(overloadSolver->ComputeSolverDiagnostics(dOverload), "Diagnostics on payload overload", testName);
        ExpectTrue(dOverload.c204OriginReturnTriggered, "Payload > 2500 kg triggers Automatic Origin Return / abort", testName);
        overloadSolver->Shutdown();

        // 4. Simultaneous GPU A/B Comparison on FieldGrid2D (TitorC204 vs Thinker::Essonne).
        auto sEssonne = ModelRegistry::CreateSolver(ModelId::ThinkerEssonne, SimulationDomain::FieldGrid2D, 1U);
        ExpectTrue(sEssonne != nullptr && sEssonne->Initialize(ctx, c204Scene), "Init Essonne overlay on C204 grid", testName);
        ExpectTrue(sEssonne->StepMultiple(5ULL, 1.0), "Step Essonne overlay", testName);
        FieldGrid2DSnapshot snapEss{};
        ExpectTrue(sEssonne->FetchFieldGrid2DSnapshot(snapEss), "Fetch Essonne overlay snapshot", testName);
        double l2Diff = 0.0;
        for (std::size_t i = 0U; i < snap.scalarField.size(); ++i) {
            const double d = snap.scalarField[i] - snapEss.scalarField[i];
            l2Diff += d * d;
        }
        l2Diff = std::sqrt(l2Diff / static_cast<double>(snap.scalarField.size()));
        ExpectTrue(std::isfinite(l2Diff) && l2Diff > 0.1, "Simultaneous C204 vs Essonne FieldGrid2D A/B residual is finite and distinct", testName);
        sEssonne->Shutdown();
        solver->Shutdown();

        // 5. RadialAndScaleProfile A/B Comparison (Thinker::TitorC204 vs Mainstream::SchwarzschildGR).
        UniverseScene profileScene{};
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "titor_c204_metric_profile.scenario.umgebung", profileScene, err),
            "Load titor_c204_metric_profile.scenario.umgebung",
            testName
        );
        auto sProfC204 = ModelRegistry::CreateSolver(ModelId::ThinkerTitorC204, SimulationDomain::RadialAndScaleProfile, 0U);
        auto sProfGR = ModelRegistry::CreateSolver(ModelId::MainstreamSchwarzschildGR, SimulationDomain::RadialAndScaleProfile, 1U);
        ExpectTrue(sProfC204 != nullptr && sProfC204->Initialize(ctx, profileScene), "Init C204 RadialScaleProfile", testName);
        ExpectTrue(sProfGR != nullptr && sProfGR->Initialize(ctx, profileScene), "Init SchwarzschildGR RadialScaleProfile", testName);

        RadialProfileSnapshot pC204{}, pGR{};
        ExpectTrue(sProfC204->FetchRadialProfileSnapshot(pC204), "Fetch C204 1D profile", testName);
        ExpectTrue(sProfGR->FetchRadialProfileSnapshot(pGR), "Fetch GR 1D profile", testName);
        ExpectTrue(pC204.xValues.size() == 256U && pGR.xValues.size() == 256U, "256 radial profile samples", testName);
        // Inner core has Negative Time g_00 > 0, outer edge at 15 m is flat Minkowski g_00 ~ -1.0 and accel < 0.05 G!
        ExpectTrue(pC204.primaryYValues.front() > 0.0, "C204 1D profile exhibits Negative Time core (g_00 > 0) near r -> 0", testName);
        ExpectTrue(std::abs(pC204.primaryYValues.back() - (-1.0)) < 0.05, "C204 1D profile screens to flat Minkowski g_00 ~ -1 at 15 m", testName);
        ExpectTrue(pC204.secondaryYValues.back() < 0.05, "C204 Van Den Broeck bubble screens exterior gravity to < 0.05 G at 15 m", testName);
        sProfC204->Shutdown();
        sProfGR->Shutdown();
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    TestVerbatimManualAndRelativityMath();

    const auto caps = umgebung::core::SystemProfiler::ProfileSystem();
    const auto plan = umgebung::core::SystemProfiler::BuildAdaptationPlan(
        caps,
        umgebung::core::HardwareAdaptationConfig{.guiActive = false}
    );

    umgebung::compute::CUDAContext ctx;
    if (!ctx.Initialize(caps, plan)) {
        std::cerr << "[FATAL] Failed to initialize CUDAContext for TitorC204Tests." << std::endl;
        return 1;
    }

    TestC204GpuLifecycleAndFailsafes(ctx, scenesDir);

    ctx.Shutdown();

    if (gFailedAssertions > 0) {
        std::cerr << "TitorC204Tests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All TitorC204Tests passed." << std::endl;
    return 0;
}
