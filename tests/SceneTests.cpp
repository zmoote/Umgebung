/**
 * @file SceneTests.cpp
 * @brief Automated CTest verification suite for all three simulation scopes:
 *        Scenarios (`scenes/scenarios/<id>.scenario.umgebung`), multi-scale
 *        Bubble Universes (`scenes/universes/<id>.umgebung`), and Elena Danaan's
 *        `FlowerOfLifeMultiverse` (`scenes/multiverses/<id>.multiverse.umgebung`).
 */

#include "umgebung/scene/FlowerOfLifeMultiverse.hpp"
#include "umgebung/scene/Multiverse.hpp"
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
     * @brief Verifies lossless JSON round-trip across all three model tiers & ScaleTier metadata.
     */
    void TestUniverseJsonRoundTrip() {
        const char* testName = "TestUniverseJsonRoundTrip";
        using namespace umgebung::scene;

        UniverseScene original{};
        original.scope = SimulationScope::BubbleUniverse;
        original.universeId = "unified_test_bubble";
        original.name = "Unified Research Test Universe";
        original.description = "Round-trip schema verification with NFW & all 5 Thinker blocks.";
        original.physicsProfile.type = PhysicsProfileType::Unified;
        original.physicsProfile.defaultModel = ModelId::UnifiedSynthesis;
        original.physicsProfile.recommendedCompareModel = "Mainstream::DarkMatterNFW";
        original.physicsProfile.integrator = IntegratorType::VelocityVerlet;
        original.physicsProfile.unitSystem = "Normalized";
        original.physicsProfile.timestep = 0.0025;
        original.physicsProfile.defaultSteps = 2000ULL;
        original.physicsProfile.mainstreamGravity.gravitationalConstant = 1.0;
        original.physicsProfile.mainstreamGravity.softeningLength = 1.0e-4;
        original.physicsProfile.darkMatterNFW.characteristicDensityRho0 = 0.03;
        original.physicsProfile.darkMatterNFW.scaleRadiusRs = 3.0;
        original.physicsProfile.unifiedField.haramein.spacetimeTorqueOmegaZ = 0.025;
        original.physicsProfile.unifiedField.pais.surfaceChargeDensity = 1.0e-3;
        original.physicsProfile.unifiedField.pais.inertialReductionCoupling = 1.0e-10;
        original.physicsProfile.unifiedField.loveBindingCoupling = 1.61803398875;
        original.physicsProfile.unifiedField.vogelWillis.driveVoltageV = 36.0;
        original.physicsProfile.unifiedField.vogelWillis.phryllCoupling = 0.05;
        original.physicsProfile.unifiedField.nonLinearTimeEnabled = true;

        BodyEntity b1{};
        b1.id = "source_crystal";
        b1.name = "Vogel Crystal Generator";
        b1.category = "PhryllCrystal";
        b1.scaleTier = ScaleTier::EntityOrDevice;
        b1.mass = 2.5;
        b1.radius = 0.1;
        b1.position = {1.25, -2.5, 0.75};
        b1.velocity = {0.1, 0.2, -0.3};
        b1.vibrationalDensity = 5.0;
        b1.consciousnessCoherence = 0.95;
        b1.phryllCharge = 12.5;
        original.bodies.push_back(b1);

        const nlohmann::json serialized = original.ToJson();
        UniverseScene restored{};
        std::string errMsg;
        ExpectTrue(UniverseScene::FromJson(serialized, restored, errMsg), "UniverseScene::FromJson succeeds", testName);
        ExpectTrue(restored.scope == SimulationScope::BubbleUniverse, "BubbleUniverse scope preserved", testName);
        ExpectTrue(restored.physicsProfile.defaultModel == ModelId::UnifiedSynthesis, "defaultModel preserved", testName);
        ExpectTrue(restored.physicsProfile.recommendedCompareModel == "Mainstream::DarkMatterNFW", "recommendedCompareModel preserved", testName);
        ExpectTrue(std::abs(restored.physicsProfile.darkMatterNFW.characteristicDensityRho0 - 0.03) < 1e-12, "NFW rho0 preserved", testName);
        ExpectTrue(std::abs(restored.physicsProfile.unifiedField.haramein.spacetimeTorqueOmegaZ - 0.025) < 1e-12, "Haramein omegaZ preserved", testName);
        ExpectTrue(std::abs(restored.physicsProfile.unifiedField.pais.inertialReductionCoupling - 1.0e-10) < 1e-20, "Pais coupling preserved", testName);
        ExpectTrue(std::abs(restored.physicsProfile.unifiedField.vogelWillis.driveVoltageV - 36.0) < 1e-12, "Vogel driveVoltageV preserved", testName);
        ExpectTrue(restored.bodies.size() == 1U && restored.bodies[0].scaleTier == ScaleTier::EntityOrDevice, "ScaleTier preserved", testName);
    }

    /**
     * @brief Verifies strict validation rejects invalid scenes.
     */
    void TestSceneValidationGuards() {
        const char* testName = "TestSceneValidationGuards";
        using namespace umgebung::scene;

        UniverseScene base{};
        base.universeId = "valid_id";
        base.name = "Valid Scene";
        base.physicsProfile.timestep = 0.01;
        BodyEntity b{};
        b.id = "b1";
        b.name = "Body 1";
        b.mass = 1.0;
        base.bodies.push_back(b);

        std::string err;
        ExpectTrue(base.Validate(err), "baseline scene validates", testName);

        { UniverseScene bad = base; bad.bodies[0].mass = -1.0; ExpectTrue(!bad.Validate(err), "negative mass rejected", testName); }
        { UniverseScene bad = base; bad.physicsProfile.timestep = 0.0; ExpectTrue(!bad.Validate(err), "zero timestep rejected", testName); }
        { UniverseScene bad = base; bad.bodies.push_back(b); ExpectTrue(!bad.Validate(err), "duplicate body IDs rejected", testName); }
        { UniverseScene bad = base; bad.schemaVersion = 99U; ExpectTrue(!bad.Validate(err), "unsupported schema_version rejected", testName); }
        { UniverseScene bad = base; bad.physicsProfile.darkMatterNFW.scaleRadiusRs = 0.0; ExpectTrue(!bad.Validate(err), "non-positive NFW r_s rejected", testName); }
    }

    /**
     * @brief Verifies loading Scenarios, Bubble Universes, and FlowerOfLifeMultiverse.
     */
    void TestThreeScopeWorkspaceLoading(const std::filesystem::path& scenesDir) {
        const char* testName = "TestThreeScopeWorkspaceLoading";
        using namespace umgebung::scene;

        // 1. Standalone Scenarios across all 3 SimulationDomains
        Multiverse scenarios;
        const std::size_t scenCount = scenarios.LoadDirectory(scenesDir / "scenarios");
        ExpectTrue(scenCount == 7U, "7 bundled standalone scenarios loaded", testName);

        bool foundGalaxy = false, foundPais = false, foundVogel = false, foundHaramein = false, foundEssonne = false;
        for (const auto& sc : scenarios.GetUniverses()) {
            ExpectTrue(sc.scope == SimulationScope::Scenario, "scenario has SimulationScope::Scenario", testName);
            if (sc.universeId == "galaxy_periphery") {
                foundGalaxy = true;
                ExpectTrue(sc.domain == SimulationDomain::NBodyDynamics, "galaxy_periphery is NBodyDynamics", testName);
            }
            if (sc.universeId == "pais_heemfg_cavity") {
                foundPais = true;
                ExpectTrue(sc.domain == SimulationDomain::FieldGrid2D, "pais_heemfg_cavity is FieldGrid2D", testName);
            }
            if (sc.universeId == "vogel_phryll_crystal") {
                foundVogel = true;
                ExpectTrue(sc.domain == SimulationDomain::FieldGrid2D, "vogel_phryll_crystal is FieldGrid2D", testName);
            }
            if (sc.universeId == "haramein_holographic_scale") {
                foundHaramein = true;
                ExpectTrue(sc.domain == SimulationDomain::RadialAndScaleProfile, "haramein_holographic_scale is RadialAndScaleProfile", testName);
            }
            if (sc.universeId == "essonne_vacuum_horizon") {
                foundEssonne = true;
                ExpectTrue(sc.domain == SimulationDomain::RadialAndScaleProfile, "essonne_vacuum_horizon is RadialAndScaleProfile", testName);
            }
        }
        ExpectTrue(foundGalaxy, "found galaxy_periphery.scenario.umgebung", testName);
        ExpectTrue(foundPais, "found pais_heemfg_cavity.scenario.umgebung (FieldGrid2D)", testName);
        ExpectTrue(foundVogel, "found vogel_phryll_crystal.scenario.umgebung (FieldGrid2D)", testName);
        ExpectTrue(foundHaramein, "found haramein_holographic_scale.scenario.umgebung", testName);
        ExpectTrue(foundEssonne, "found essonne_vacuum_horizon.scenario.umgebung", testName);

        // 2. Multi-scale Bubble Universes with domain-appropriate sub_simulations
        Multiverse universes;
        const std::size_t univCount = universes.LoadDirectory(scenesDir / "universes");
        ExpectTrue(univCount == 2U, "2 bundled Bubble Universes loaded", testName);

        bool foundSolPrime = false;
        for (const auto& u : universes.GetUniverses()) {
            ExpectTrue(u.scope == SimulationScope::BubbleUniverse, "universe has SimulationScope::BubbleUniverse", testName);
            if (u.universeId == "sol_prime") {
                foundSolPrime = true;
                ExpectTrue(u.bodies.size() == 3U, "sol_prime keeps only 3 macroscopic N-body objects in bodies[]", testName);
                ExpectTrue(u.subSimulations.size() == 3U, "sol_prime delegates 3 sub-scale tiers via sub_simulations[]", testName);
                bool hasFieldSub = false, hasSubatomicSub = false;
                for (const auto& sub : u.subSimulations) {
                    if (sub.domain == SimulationDomain::FieldGrid2D) { hasFieldSub = true; }
                    if (sub.scaleTier == ScaleTier::SubatomicPSU &&
                        sub.domain == SimulationDomain::RadialAndScaleProfile) { hasSubatomicSub = true; }
                }
                ExpectTrue(hasFieldSub, "sol_prime delegates Vogel/Pais devices to FieldGrid2D", testName);
                ExpectTrue(hasSubatomicSub, "sol_prime delegates SubatomicPSU proton to RadialAndScaleProfile", testName);
            }
        }
        ExpectTrue(foundSolPrime, "found sol_prime.umgebung Bubble Universe", testName);

        // 3. Flower of Life Multiverse geometry & manifest loading
        const auto seedNodes = FlowerOfLifeMultiverse::GenerateCanonicalLattice(2.0, 1U);
        ExpectTrue(seedNodes.size() == 7U, "Seed of Life (maxRing=1) generates 7 spheres", testName);
        for (std::size_t k = 1U; k <= 6U; ++k) {
            const double distOrigin = seedNodes[k].latticePosition.Norm();
            ExpectTrue(std::abs(distOrigin - 2.0) < 1e-12, "Ring-1 node sits at exact distance R from origin", testName);
            const std::size_t next = (k == 6U) ? 1U : (k + 1U);
            const double distNeighbor = (seedNodes[k].latticePosition - seedNodes[next].latticePosition).Norm();
            ExpectTrue(std::abs(distNeighbor - 2.0) < 1e-12, "Adjacent Ring-1 nodes sit at exact distance R", testName);
        }

        const auto flowerNodes = FlowerOfLifeMultiverse::GenerateCanonicalLattice(1.0, 2U);
        ExpectTrue(flowerNodes.size() == 19U, "Flower of Life (maxRing=2) generates 19 spheres", testName);

        FlowerOfLifeMultiverse mv;
        std::string err;
        ExpectTrue(
            mv.LoadFromFile(scenesDir / "multiverses" / "flower_of_life_default.multiverse.umgebung", err),
            "Load flower_of_life_default.multiverse.umgebung",
            testName
        );
        ExpectTrue(mv.GetNodes().size() == 19U, "Loaded manifest has 19 Flower of Life nodes", testName);
        std::size_t populatedCount = 0U;
        for (const auto& n : mv.GetNodes()) {
            if (n.loadedUniverse.has_value()) {
                ++populatedCount;
            }
        }
        ExpectTrue(populatedCount == 2U, "Flower of Life manifest resolved 2 linked Bubble Universes", testName);
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    TestUniverseJsonRoundTrip();
    TestSceneValidationGuards();
    TestThreeScopeWorkspaceLoading(scenesDir);

    if (gFailedAssertions > 0) {
        std::cerr << "SceneTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All SceneTests passed." << std::endl;
    return 0;
}
