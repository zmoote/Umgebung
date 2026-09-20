#include <iostream>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include "umgebung/engine.hpp"
#include "umgebung/hierarchy.hpp"
#include "umgebung/types.hpp"

using namespace umgebung;

int main(int argc, char** argv) {
    cxxopts::Options options("umgebung-cli", "Headless execution interface for the Umgebung Reality Engine");

    options.add_options()
        ("l,levels", "Number of Flower of Life geometry levels to generate", cxxopts::value<int>()->default_value("1"))
        ("h,help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int levels = result["levels"].as<int>();

    if (levels < 0) {
        fmt::print(stderr, "Error: Levels cannot be negative.\n");
        return 1;
    }

    fmt::print("Initializing Umgebung Reality Framework...\n\n");
    
    // 1. Create the Multiverse (Top of the hierarchy)
    Multiverse the_multiverse("The Grand Multiverse");
    
    // 2. Create an Observer initially zoomed out to the Multiverse scale
    Observer player_camera({0.0, 0.0, 0.0, ScaleLevel::Multiverse}, ScaleLevel::Multiverse);
    
    // Update multiverse based on observer (Should trigger load)
    the_multiverse.update(player_camera);

    fmt::print("\n--- Observer Zooming in to Universe Scale ---\n");
    player_camera.focus_scale = ScaleLevel::Universe;
    the_multiverse.update(player_camera);

    fmt::print("\n--- Observer Zooming in to Galaxy Scale ---\n");
    player_camera.focus_scale = ScaleLevel::Galaxy;
    the_multiverse.update(player_camera);

    fmt::print("\n--- Observer Zooming in to StarSystem Scale ---\n");
    player_camera.focus_scale = ScaleLevel::StarSystem;
    the_multiverse.update(player_camera);

    fmt::print("\n--- Observer Flying to Earth and Zooming to Planet Scale ---\n");
    player_camera.position.x = 1.5e11; // Teleport to Earth's coordinate in the solar system
    player_camera.focus_scale = ScaleLevel::Planet;
    the_multiverse.update(player_camera);

    fmt::print("\n--- Observer Zooming in to Quantum Scale ---\n");
    player_camera.focus_scale = ScaleLevel::Quantum;
    the_multiverse.update(player_camera);
    
    fmt::print("\n================================================\n");
    fmt::print("Generating Flower of Life PSU geometry at Quantum Scale (Levels: {})\n", levels);
    
    FlowerOfLife engine;
    engine.generate(levels);

    const auto& units = engine.getUnits();
    
    fmt::print("Generation complete.\n");
    fmt::print("Total PSUs calculated: {}\n", units.size());
    
    if (!units.empty()) {
        fmt::print("Center PSU coordinates: ({}, {}, {})\n", 
            units[0].getCenter().x, units[0].getCenter().y, units[0].getCenter().z);
        if (units.size() > 1) {
            fmt::print("Outer PSU coordinates: ({}, {}, {})\n", 
                units.back().getCenter().x, units.back().getCenter().y, units.back().getCenter().z);
        }
    }

    return 0;
}
