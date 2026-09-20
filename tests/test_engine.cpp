#include <gtest/gtest.h>
#include "umgebung/engine.hpp"

using namespace umgebung;

TEST(EngineTest, PlanckLengthConstants) {
    EXPECT_DOUBLE_EQ(constants::PLANCK_LENGTH, 1.616255e-35);
    EXPECT_DOUBLE_EQ(constants::PI, 3.14159265358979323846);
}

TEST(EngineTest, PSURadiusAndVolume) {
    PSU psu;
    double expected_radius = constants::PLANCK_LENGTH / 2.0;
    EXPECT_DOUBLE_EQ(PSU::radius(), expected_radius);

    double expected_volume = (4.0 / 3.0) * constants::PI * expected_radius * expected_radius * expected_radius;
    EXPECT_DOUBLE_EQ(PSU::volume(), expected_volume);
}

TEST(EngineTest, FlowerOfLifeGeneration_Level1) {
    FlowerOfLife fol;
    fol.generate(1);
    
    // Center PSU + 6 PSUs in the first ring = 7 PSUs total
    const auto& units = fol.getUnits();
    EXPECT_EQ(units.size(), 7);

    // Verify center is at origin
    EXPECT_DOUBLE_EQ(units[0].getCenter().x, 0.0);
    EXPECT_DOUBLE_EQ(units[0].getCenter().y, 0.0);
    EXPECT_DOUBLE_EQ(units[0].getCenter().z, 0.0);
}

TEST(EngineTest, FlowerOfLifeGeneration_Level2) {
    FlowerOfLife fol;
    fol.generate(2);
    
    // Level 1: 6 PSUs
    // Level 2: 12 PSUs
    // Total = 1 + 6 + 12 = 19
    const auto& units = fol.getUnits();
    EXPECT_EQ(units.size(), 19);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

