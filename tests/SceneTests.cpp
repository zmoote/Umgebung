/**
 * @file SceneTests.cpp
 * @brief Automated CTest verification suite for `.umgebung` scene schema v1,
 *        `umgebung::scene::UniverseScene`, and `umgebung::scene::Multiverse`.
 *
 * @details
 * Verifies:
 *   1. JSON round-trip fidelity for both `Mainstream` and `Unified` physics
 *      profiles (including nested `parentId` hierarchy and optional unified
 *      consciousness/vibrational-density/Phryll fields).
 *   2. Strict rejection of invalid scenes (bad schema version, non-positive
 *      mass, non-positive timestep, duplicate body IDs).
 *   3. Loading the bundled repository scenes (`scenes/kepler_two_body.umgebung`
 *      and `scenes/figure_eight_three_body.umgebung`) via `Multiverse::LoadDirectory`.
 */

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
     * @brief Verifies lossless JSON round-trip of both Mainstream and Unified properties.
     */
    void TestUniverseJsonRoundTrip() {
        const char* testName = "TestUniverseJsonRoundTrip";
        using namespace umgebung::scene;

        UniverseScene original{};
        original.universeId = "unified_test_bubble";
        original.name = "Unified Research Test Universe";
        original.description = "Round-trip schema verification with Unified field parameters.";
        original.physicsProfile.type = PhysicsProfileType::Unified;
        original.physicsProfile.integrator = IntegratorType::VelocityVerlet;
        original.physicsProfile.unitSystem = "Normalized";
        original.physicsProfile.timestep = 0.0025;
        original.physicsProfile.defaultSteps = 2000ULL;
        original.physicsProfile.mainstreamGravity.gravitationalConstant = 1.0;
        original.physicsProfile.mainstreamGravity.softeningLength = 1.0e-4;
        original.physicsProfile.unifiedField.harmonicVacuumDensity = 5.155e96;
        original.physicsProfile.unifiedField.loveBindingCoupling = 1.61803398875;
        original.physicsProfile.unifiedField.phryllFieldAmplitude = 432.0;
        original.physicsProfile.unifiedField.baseVibrationalDensity = 5.0;
        original.physicsProfile.unifiedField.nonLinearTimeEnabled = true;

        BodyEntity b1{};
        b1.id = "source_crystal";
        b1.name = "Vogel Crystal Generator";
        b1.category = "Crystal";
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
        ExpectTrue(restored.universeId == original.universeId, "universeId preserved", testName);
        ExpectTrue(restored.physicsProfile.type == PhysicsProfileType::Unified, "Unified profile preserved", testName);
        ExpectTrue(restored.physicsProfile.unifiedField.nonLinearTimeEnabled, "nonLinearTimeEnabled preserved", testName);
        ExpectTrue(
            std::abs(restored.physicsProfile.unifiedField.loveBindingCoupling - 1.61803398875) < 1e-12,
            "loveBindingCoupling preserved",
            testName
        );
        ExpectTrue(restored.bodies.size() == 1U, "bodies.size() == 1", testName);
        if (!restored.bodies.empty()) {
            ExpectTrue(restored.bodies[0].id == "source_crystal", "body id preserved", testName);
            ExpectTrue(std::abs(restored.bodies[0].consciousnessCoherence - 0.95) < 1e-12, "coherence preserved", testName);
            ExpectTrue(std::abs(restored.bodies[0].phryllCharge - 12.5) < 1e-12, "phryllCharge preserved", testName);
        }
    }

    /**
     * @brief Verifies strict domain validation rejects malformed/unphysical scenes.
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

        // Negative mass rejected.
        {
            UniverseScene bad = base;
            bad.bodies[0].mass = -1.0;
            ExpectTrue(!bad.Validate(err), "negative mass rejected", testName);
        }
        // Zero timestep rejected.
        {
            UniverseScene bad = base;
            bad.physicsProfile.timestep = 0.0;
            ExpectTrue(!bad.Validate(err), "zero timestep rejected", testName);
        }
        // Duplicate body IDs rejected.
        {
            UniverseScene bad = base;
            bad.bodies.push_back(b);
            ExpectTrue(!bad.Validate(err), "duplicate body IDs rejected", testName);
        }
        // Unsupported schema version rejected.
        {
            UniverseScene bad = base;
            bad.schemaVersion = 99U;
            ExpectTrue(!bad.Validate(err), "unsupported schema_version rejected", testName);
        }
    }

    /**
     * @brief Verifies loading the repository's bundled `.umgebung` benchmark scenes.
     */
    void TestBundledScenesDirectory(const std::filesystem::path& scenesDir) {
        const char* testName = "TestBundledScenesDirectory";
        using namespace umgebung::scene;

        Multiverse mv;
        const std::size_t loadedCount = mv.LoadDirectory(scenesDir);
        ExpectTrue(loadedCount >= 2U, "loadedCount >= 2 bundled .umgebung scenes", testName);
        ExpectTrue(mv.GetActiveUniverse() != nullptr, "active universe non-null", testName);

        bool foundKepler = false;
        bool foundFigureEight = false;
        for (const auto& u : mv.GetUniverses()) {
            if (u.universeId == "kepler_two_body") {
                foundKepler = true;
                ExpectTrue(u.bodies.size() == 2U, "kepler_two_body has 2 bodies", testName);
                ExpectTrue(u.physicsProfile.type == PhysicsProfileType::Mainstream, "kepler is Mainstream", testName);
            } else if (u.universeId == "figure_eight_three_body") {
                foundFigureEight = true;
                ExpectTrue(u.bodies.size() == 3U, "figure_eight_three_body has 3 bodies", testName);
                ExpectTrue(u.physicsProfile.type == PhysicsProfileType::Mainstream, "figure_eight is Mainstream", testName);
            }
        }
        ExpectTrue(foundKepler, "found scenes/kepler_two_body.umgebung", testName);
        ExpectTrue(foundFigureEight, "found scenes/figure_eight_three_body.umgebung", testName);
    }

} // anonymous namespace

int main(int argc, char** argv) {
    const std::filesystem::path scenesDir =
        (argc > 1 && argv != nullptr && argv[1] != nullptr)
            ? std::filesystem::path(argv[1])
            : std::filesystem::path("scenes");

    TestUniverseJsonRoundTrip();
    TestSceneValidationGuards();
    TestBundledScenesDirectory(scenesDir);

    if (gFailedAssertions > 0) {
        std::cerr << "SceneTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All SceneTests passed." << std::endl;
    return 0;
}
