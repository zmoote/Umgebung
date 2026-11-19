/**
 * @file NewtonianNBodyTests.cpp
 * @brief Automated CTest physics verification suite for Pillar 1's FP64 CUDA
 *        Newtonian N-body solver (`umgebung::models::mainstream::NewtonianNBodySolver`).
 *
 * @details
 * Executes five rigorous computational physics verification benchmarks live on
 * the host NVIDIA GPU:
 *   1. **Analytical Kepler Two-Body Circular Orbit (10 Full Orbits, $T = 20\pi$):**
 *      Verifies linear momentum conservation ($< 10^{-12}$), angular momentum
 *      conservation ($< 10^{-11}$), symplectic total energy conservation
 *      ($|\Delta E / E_0| < 10^{-7}$), and analytical trajectory closure.
 *   2. **Second-Order Convergence Verification ($O(\Delta t^2)$):**
 *      Halves $\Delta t$ on the exact Kepler orbit and confirms position error
 *      shrinks by $\approx 4\times$ ($3.5 \le e(\Delta t) / e(\Delta t / 2) \le 4.5$).
 *   3. **Symplectic Time-Reversibility Verification:**
 *      Integrates a 3-body system forward $K$ steps ($+\Delta t$) and backward
 *      $K$ steps ($-\Delta t$), verifying exact phase-space recovery ($< 10^{-11}$).
 *   4. **Chenciner-Montgomery Figure-8 Three-Body Choreography (1 Full Period):**
 *      Loads `scenes/figure_eight_three_body.umgebung`, integrates $T = 6.32591398$
 *      on the GPU, and verifies energy, linear/angular momentum, and periodic return.
 *   5. **Multi-Block Shared-Memory Tiling Stress Test ($N = 1024$ bodies = 4 CUDA blocks):**
 *      Exercises multi-tile `__shared__` memory sweeps across 4 blocks of 256
 *      threads, verifying linear & angular momentum conservation ($< 10^{-11}$)
 *      and bounded symplectic energy.
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/mainstream/NewtonianNBodySolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numbers>
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
     * @brief Test 1: Analytical Kepler Two-Body Circular Orbit over 10 full orbits.
     */
    void TestKeplerTwoBodyConservationAndTrajectory(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestKeplerTwoBodyConservationAndTrajectory";
        using namespace umgebung::scene;
        using namespace umgebung::models::mainstream;

        UniverseScene kepler{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "kepler_two_body.scenario.umgebung", kepler, err),
            "Load scenarios/kepler_two_body.scenario.umgebung",
            testName
        );

        NewtonianNBodySolver solver;
        ExpectTrue(solver.Initialize(ctx, kepler), "solver.Initialize(kepler)", testName);

        NBodyDiagnostics d0{};
        ExpectTrue(solver.ComputeDiagnostics(d0), "initial diagnostics", testName);

        // Integrate exactly 10 full orbital periods: T_total = 10 * 2*pi.
        constexpr std::uint64_t kStepsPerOrbit = 2000ULL;
        constexpr std::uint64_t kOrbits = 10ULL;
        constexpr std::uint64_t kTotalSteps = kStepsPerOrbit * kOrbits;
        const double dt = (2.0 * std::numbers::pi) / static_cast<double>(kStepsPerOrbit);

        ExpectTrue(solver.StepMultiple(kTotalSteps, dt), "StepMultiple 10 Kepler orbits", testName);

        NBodyDiagnostics df{};
        ExpectTrue(solver.ComputeDiagnostics(df), "final diagnostics", testName);

        std::vector<BodyEntity> finalBodies = kepler.bodies;
        ExpectTrue(solver.SyncToHost(finalBodies), "SyncToHost", testName);

        const double momDrift = (df.totalLinearMomentum - d0.totalLinearMomentum).Norm();
        const double angMomDrift = (df.totalAngularMomentum - d0.totalAngularMomentum).Norm();
        const double posErrPlanet = (finalBodies[1].position - kepler.bodies[1].position).Norm();

        std::cout << "[Kepler 10-Orbit] relEnergyDrift=" << df.relativeEnergyDrift
                  << " | momDrift=" << momDrift
                  << " | angMomDrift=" << angMomDrift
                  << " | returnPosErr=" << posErrPlanet << std::endl;

        ExpectTrue(momDrift < 1.0e-12, "linear momentum conserved < 1e-12", testName);
        ExpectTrue(angMomDrift < 1.0e-11, "angular momentum conserved < 1e-11", testName);
        ExpectTrue(df.relativeEnergyDrift < 1.0e-7, "symplectic energy drift < 1e-7", testName);
        ExpectTrue(posErrPlanet < 5.0e-4, "planet returns to analytical 10-orbit position < 5e-4", testName);

        solver.Shutdown();
    }

    /**
     * @brief Test 2: Verifies 2nd-order convergence $O(\Delta t^2)$ of Velocity Verlet.
     */
    void TestSecondOrderConvergence(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestSecondOrderConvergence";
        using namespace umgebung::scene;
        using namespace umgebung::models::mainstream;

        UniverseScene kepler{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "kepler_two_body.scenario.umgebung", kepler, err),
            "Load scenarios/kepler_two_body.scenario.umgebung",
            testName
        );

        auto runOneOrbitAndGetPosError = [&](std::uint64_t steps) -> double {
            NewtonianNBodySolver solver;
            if (!solver.Initialize(ctx, kepler)) {
                return -1.0;
            }
            const double dt = (2.0 * std::numbers::pi) / static_cast<double>(steps);
            if (!solver.StepMultiple(steps, dt)) {
                return -1.0;
            }
            std::vector<BodyEntity> out = kepler.bodies;
            if (!solver.SyncToHost(out)) {
                return -1.0;
            }
            solver.Shutdown();
            return (out[1].position - kepler.bodies[1].position).Norm();
        };

        const double errCoarse = runOneOrbitAndGetPosError(500ULL);
        const double errFine = runOneOrbitAndGetPosError(1000ULL);
        const double ratio = (errFine > 0.0) ? (errCoarse / errFine) : 0.0;

        std::cout << "[Convergence O(dt^2)] err(500)=" << errCoarse
                  << " | err(1000)=" << errFine
                  << " | ratio=" << ratio << " (expected ~4.0)" << std::endl;

        ExpectTrue(errCoarse > 0.0 && errFine > 0.0, "positive finite errors", testName);
        ExpectTrue(ratio > 3.8 && ratio < 4.2, "2nd-order error ratio in [3.8, 4.2]", testName);
    }

    /**
     * @brief Test 3: Verifies exact time-reversibility of Velocity Verlet on a 3-body system.
     */
    void TestTimeReversibility(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestTimeReversibility";
        using namespace umgebung::scene;
        using namespace umgebung::models::mainstream;

        UniverseScene fig8{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "figure_eight_three_body.scenario.umgebung", fig8, err),
            "Load scenarios/figure_eight_three_body.scenario.umgebung",
            testName
        );

        NewtonianNBodySolver solver;
        ExpectTrue(solver.Initialize(ctx, fig8), "solver.Initialize(fig8)", testName);

        constexpr std::uint64_t kSteps = 1000ULL;
        constexpr double kDt = 0.001;

        // Integrate forward +dt, then reverse exactly -dt for the same number of steps.
        ExpectTrue(solver.StepMultiple(kSteps, +kDt), "forward integration +dt", testName);
        ExpectTrue(solver.StepMultiple(kSteps, -kDt), "backward integration -dt", testName);

        std::vector<BodyEntity> recovered = fig8.bodies;
        ExpectTrue(solver.SyncToHost(recovered), "SyncToHost after reversal", testName);

        double maxPosErr = 0.0;
        double maxVelErr = 0.0;
        for (std::size_t i = 0U; i < fig8.bodies.size(); ++i) {
            maxPosErr = std::max(maxPosErr, (recovered[i].position - fig8.bodies[i].position).Norm());
            maxVelErr = std::max(maxVelErr, (recovered[i].velocity - fig8.bodies[i].velocity).Norm());
        }

        std::cout << "[Time Reversibility] maxPosErr=" << maxPosErr
                  << " | maxVelErr=" << maxVelErr << std::endl;

        ExpectTrue(maxPosErr < 1.0e-11, "position recovered < 1e-11", testName);
        ExpectTrue(maxVelErr < 1.0e-11, "velocity recovered < 1e-11", testName);

        solver.Shutdown();
    }

    /**
     * @brief Test 4: Chenciner-Montgomery Figure-8 Three-Body Periodic Choreography.
     */
    void TestFigureEightThreeBodyChoreography(
        umgebung::compute::CUDAContext& ctx,
        const std::filesystem::path& scenesDir
    ) {
        const char* testName = "TestFigureEightThreeBodyChoreography";
        using namespace umgebung::scene;
        using namespace umgebung::models::mainstream;

        UniverseScene fig8{};
        std::string err;
        ExpectTrue(
            UniverseScene::LoadFromFile(scenesDir / "scenarios" / "figure_eight_three_body.scenario.umgebung", fig8, err),
            "Load scenarios/figure_eight_three_body.scenario.umgebung",
            testName
        );

        NewtonianNBodySolver solver;
        ExpectTrue(solver.Initialize(ctx, fig8), "solver.Initialize(fig8)", testName);

        NBodyDiagnostics d0{};
        ExpectTrue(solver.ComputeDiagnostics(d0), "initial fig8 diagnostics", testName);

        // Exact published Figure-8 choreography period T = 6.32591398.
        constexpr double kPeriod = 6.32591398;
        constexpr std::uint64_t kSteps = 12000ULL;
        const double dt = kPeriod / static_cast<double>(kSteps);

        ExpectTrue(solver.StepMultiple(kSteps, dt), "StepMultiple 1 full Figure-8 period", testName);

        NBodyDiagnostics df{};
        ExpectTrue(solver.ComputeDiagnostics(df), "final fig8 diagnostics", testName);

        std::vector<BodyEntity> finalBodies = fig8.bodies;
        ExpectTrue(solver.SyncToHost(finalBodies), "SyncToHost fig8", testName);

        double maxReturnErr = 0.0;
        for (std::size_t i = 0U; i < fig8.bodies.size(); ++i) {
            maxReturnErr = std::max(maxReturnErr, (finalBodies[i].position - fig8.bodies[i].position).Norm());
        }

        const double momDrift = (df.totalLinearMomentum - d0.totalLinearMomentum).Norm();
        const double angMomDrift = (df.totalAngularMomentum - d0.totalAngularMomentum).Norm();

        std::cout << "[Figure-8 3-Body] relEnergyDrift=" << df.relativeEnergyDrift
                  << " | momDrift=" << momDrift
                  << " | angMomDrift=" << angMomDrift
                  << " | maxReturnErr=" << maxReturnErr << std::endl;

        ExpectTrue(momDrift < 1.0e-12, "Figure-8 linear momentum < 1e-12", testName);
        ExpectTrue(angMomDrift < 1.0e-12, "Figure-8 angular momentum < 1e-12", testName);
        ExpectTrue(df.relativeEnergyDrift < 1.0e-6, "Figure-8 energy drift < 1e-6", testName);
        ExpectTrue(maxReturnErr < 1.0e-3, "Figure-8 periodic return error < 1e-3", testName);

        solver.Shutdown();
    }

    /**
     * @brief Test 5: Multi-Block Shared-Memory Tiling Stress Test (N = 1024 bodies = 4 CUDA blocks).
     */
    void TestMultiBlockSharedMemoryTiling(umgebung::compute::CUDAContext& ctx) {
        const char* testName = "TestMultiBlockSharedMemoryTiling";
        using namespace umgebung::scene;
        using namespace umgebung::models::mainstream;

        constexpr std::size_t kBodyCount = 1024U; // 4 full 256-thread shared-memory tiles.

        UniverseScene cluster{};
        cluster.universeId = "multiblock_cluster_1024";
        cluster.name = "1024-Body Deterministic Softened Cluster";
        cluster.physicsProfile.type = PhysicsProfileType::Mainstream;
        cluster.physicsProfile.integrator = IntegratorType::VelocityVerlet;
        cluster.physicsProfile.timestep = 0.005;
        cluster.physicsProfile.mainstreamGravity.gravitationalConstant = 1.0;
        cluster.physicsProfile.mainstreamGravity.softeningLength = 0.15;

        // Deterministic quasi-spherical symmetric distribution (zero net momentum).
        cluster.bodies.reserve(kBodyCount);
        for (std::size_t i = 0U; i < kBodyCount; ++i) {
            const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(kBodyCount);
            const double layer = 0.5 + 1.5 * static_cast<double>(i % 16U) / 15.0;
            const double zVal = -0.5 + static_cast<double>((i / 16U) % 16U) / 15.0;

            BodyEntity b{};
            b.id = "body_" + std::to_string(i);
            b.name = b.id;
            b.mass = 1.0 / static_cast<double>(kBodyCount);
            b.position = {layer * std::cos(angle), layer * std::sin(angle), zVal};
            b.velocity = {-0.25 * std::sin(angle), 0.25 * std::cos(angle), 0.0};
            cluster.bodies.push_back(std::move(b));
        }

        NewtonianNBodySolver solver;
        ExpectTrue(solver.Initialize(ctx, cluster), "solver.Initialize(1024 bodies)", testName);

        NBodyDiagnostics d0{};
        ExpectTrue(solver.ComputeDiagnostics(d0), "1024-body initial diagnostics", testName);

        ExpectTrue(solver.StepMultiple(100ULL, 0.005), "StepMultiple 100 steps for 1024 bodies", testName);

        NBodyDiagnostics df{};
        ExpectTrue(solver.ComputeDiagnostics(df), "1024-body final diagnostics", testName);

        const double momDrift = (df.totalLinearMomentum - d0.totalLinearMomentum).Norm();
        const double angMomDrift = (df.totalAngularMomentum - d0.totalAngularMomentum).Norm();

        std::cout << "[Multi-Block 1024-Body] relEnergyDrift=" << df.relativeEnergyDrift
                  << " | momDrift=" << momDrift
                  << " | angMomDrift=" << angMomDrift << std::endl;

        ExpectTrue(momDrift < 1.0e-11, "1024-body linear momentum conserved < 1e-11", testName);
        ExpectTrue(angMomDrift < 1.0e-11, "1024-body angular momentum conserved < 1e-11", testName);
        ExpectTrue(df.relativeEnergyDrift < 1.0e-4, "1024-body symplectic energy drift < 1e-4", testName);

        solver.Shutdown();
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    const auto caps = umgebung::core::SystemProfiler::ProfileSystem();
    const auto plan = umgebung::core::SystemProfiler::BuildAdaptationPlan(
        caps,
        umgebung::core::HardwareAdaptationConfig{.guiActive = false}
    );

    umgebung::compute::CUDAContext ctx;
    if (!ctx.Initialize(caps, plan)) {
        std::cerr << "[FATAL] Failed to initialize CUDAContext for NewtonianNBodyTests." << std::endl;
        return 1;
    }

    TestKeplerTwoBodyConservationAndTrajectory(ctx, scenesDir);
    TestSecondOrderConvergence(ctx, scenesDir);
    TestTimeReversibility(ctx, scenesDir);
    TestFigureEightThreeBodyChoreography(ctx, scenesDir);
    TestMultiBlockSharedMemoryTiling(ctx);

    ctx.Shutdown();

    if (gFailedAssertions > 0) {
        std::cerr << "NewtonianNBodyTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All NewtonianNBodyTests passed." << std::endl;
    return 0;
}
