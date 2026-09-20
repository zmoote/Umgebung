#include "umgebung/hierarchy.hpp"
#include <fmt/core.h>

namespace umgebung {

    void Multiverse::load() {
        fmt::print("[Procedural Generation] Multiverse '{}' expanding...\n", name);
        // Multiverses are probabilistic. We generate a handful of Universes for the observer.
        // E.g., The "Prime" Universe and adjacent parallel dimensions.
        children.push_back(std::make_shared<Universe>("Prime Universe", Coordinate{0, 0, 0, ScaleLevel::Universe}));
        children.push_back(std::make_shared<Universe>("Mirror Universe", Coordinate{1e26, 0, 0, ScaleLevel::Universe}));
        is_loaded = true;
    }

    void Universe::load() {
        fmt::print("[Procedural Generation] Universe '{}' resolving galaxies...\n", name);
        // Generate abstract representations of 200 Billion Galaxies
        // We only instantiate a few for demonstration in the local cluster
        children.push_back(std::make_shared<Galaxy>("Milky Way", Coordinate{0, 0, 0, ScaleLevel::Galaxy}));
        children.push_back(std::make_shared<Galaxy>("Andromeda", Coordinate{2.5e22, 0, 0, ScaleLevel::Galaxy}));
        is_loaded = true;
    }

    void Galaxy::load() {
        fmt::print("[Procedural Generation] Galaxy '{}' generating stellar sectors...\n", name);
        // Instantiates local star systems
        children.push_back(std::make_shared<StarSystem>("Sol", Coordinate{0, 0, 0, ScaleLevel::StarSystem}));
        children.push_back(std::make_shared<StarSystem>("Alpha Centauri", Coordinate{4e16, 0, 0, ScaleLevel::StarSystem}));
        is_loaded = true;
    }

    void StarSystem::load() {
        fmt::print("[Procedural Generation] Star System '{}' forming planetary bodies...\n", name);
        children.push_back(std::make_shared<Planet>("Earth", Coordinate{1.5e11, 0, 0, ScaleLevel::Planet}, VibrationalState(3.0)));
        children.push_back(std::make_shared<Planet>("Mars", Coordinate{2.2e11, 0, 0, ScaleLevel::Planet}, VibrationalState(3.0)));
        is_loaded = true;
    }

    void Planet::load() {
        fmt::print("[Procedural Generation] Planet '{}' zooming into Quantum Realm (Density: {:.1f})...\n", name, vib_state.density);
        // Switch from Classical/Macro generation to Micro generation
        children.push_back(std::make_shared<QuantumRegion>("Local Quantum Field", Coordinate{0,0,0, ScaleLevel::Quantum}));
        is_loaded = true;
    }

    void QuantumRegion::load() {
        fmt::print("[Procedural Generation] Quantum Region '{}' reaching Planck scale.\n", name);
        // At this scale, the CPU passes control over to the CUDA Engine to map the PSUs
        fmt::print("-> Invoking CUDA Engine to calculate Planck Spherical Units (PSUs)...\n");
        // This is where umgebung::FlowerOfLife::generate() would be dynamically attached to the region
        is_loaded = true;
    }

} // namespace umgebung
