# Model Concepts for Umgebung

This document defines the core concepts modeled in *Umgebung*, a C++ and CUDA project representing ideas from the UFO Disclosure Community. These concepts shape the `Umgebung::Model` namespace, including classes like `Universe`, `Galaxy`, and `Hierarchy`, and inform rendering and physics subsystems. Citations provide traceability and credit to original sources.

## 1. Multiverse Hierarchy
The multiverse is a fractal, interconnected structure of universes, represented as spheres with intersection nodes for navigation, centered around an infinite singularity.

- **Definition**: A graph-like structure where each universe (sphere) contains galaxies and is linked via nodes representing dimensional or temporal transitions. The structure is fractal, with universes nested within an infinite template, converging at a central singularity (Source).
- **Implementation**: The `Umgebung::Model::Hierarchy` class uses `nlohmann::json` to load a graph of universes from JSON files, with `std::vector<Umgebung::Model::Universe>` storing universe data. Navigation methods like `navigate_to(node_id)` use intersection nodes for Stargate-like jumps. The fractal structure is modeled as a voxel grid, using Haramein’s Planck spherical unit (PSU) tiling, visualized as a 3D Flower of Life (FoL) lattice.
- **Sources**:
  - Danaan (2025) describes the multiverse as a three-dimensional holographic structure of spheres, each a universe, merging into a central singularity. Navigation uses frequency-based keys and intersection nodes ([Danaan, 2025a](#danaan-2025a)).
  - Haramein (2013) models spacetime using Planck spherical units (PSUs) with Planck length diameters, tiling volumes and horizons to derive mass and gravity. This informs the voxel-based node connections in `Hierarchy` and CUDA-accelerated simulations ([Haramein, 2013](#haramein-2013)).
  - Resonance Science Foundation (2024) describes the 3D Flower of Life as a visualization of PSU tiling, encoding gravitational fields for objects like protons or galaxies, applied to `Hierarchy`’s voxel grid and `View` rendering ([RSF, 2024](#rsf-2024)).

## 2. Density and Consciousness
Densities represent qualitative stages of consciousness or existence, with souls incarnated across beings, influencing their placement in the multiverse.

- **Definition**: Discrete levels (e.g., 3rd to 12th, with 13th as Source Creator) representing evolutionary states of consciousness, not tied to numerical frequencies. All beings have a soul, shaping their interaction with the multiverse.
- **Implementation**: The `Umgebung::Model::Universe` class includes an `int density_level_` attribute to categorize universes. The `Umgebung::Model::Galaxy` class includes a `soul_influence_` attribute (e.g., `float`) to model consciousness effects. The `Hierarchy` class filters navigation by density.
- **Sources**:
  - Danaan (2025) defines density as stages of consciousness, with all beings possessing a soul, influencing the `density_level_` attribute ([Danaan, 2025b](#danaan-2025b)).
  - Danaan (2025) notes the 13th density as the Source Creator, shaping the central singularity in `Hierarchy` ([Danaan, 2025a](#danaan-2025a)).

## 3. Universes and Galaxies
Universes are containers for galaxies, modeled as spheres with physical and metaphysical properties, supporting fractal interactions.

- **Definition**: Universes are spherical entities containing galaxies, each with spatial coordinates (`glm::vec3`), mass (`float`), and density. Galaxies interact via physical forces (e.g., gravity) and metaphysical connections (e.g., soul or frequency-based).
- **Implementation**: The `Umgebung::Model::Universe` class holds a `std::vector<Galaxy>`, with each `Galaxy` storing position, mass, `soul_influence_`, and `magic_influence_`. The `Umgebung::Physics::PhysicsEngine` uses PhysX for CPU-based interactions, and `CUDAKernels.cu` computes holographic gravitational forces using Haramein’s PSU tiling.
- **Sources**:
  - Haramein (2013) derives gravitational fields using PSU tiling of spherical volumes and horizons, applied to protons and black holes. This guides CUDA-accelerated simulations in `CUDAKernels.cu` using voxelized spheres ([Haramein, 2013](#haramein-2013)).
  - Danaan (2025) portrays universes as holographic spheres with fractal interrelations, influencing `Universe` and `Galaxy` designs ([Danaan, 2025a](#danaan-2025a)).

## 4. Navigation and Translocation
Navigation through the multiverse uses intersection nodes and frequency-based keys, akin to Stargates, with teleportation and bilocation as distinct mechanisms.

- **Definition**: Navigation involves jumping between universes via nodes, using frequency-based keys. Teleportation transfers physical bodies, while bilocation projects consciousness ethereally (astral travel).
- **Implementation**: The `Umgebung::Model::Hierarchy::navigate_to(node_id, frequency_key)` method simulates Stargate-like jumps, using a `std::string` or `float` for frequency keys. Bilocation is visualized in `Umgebung::Render::View` as ethereal projections (e.g., translucent shaders).
- **Sources**:
  - Danaan (2025) explains navigation via intersection nodes and frequency keys, shaping `Hierarchy` navigation logic ([Danaan, 2025a](#danaan-2025a)).
  - Danaan (2025) distinguishes teleportation (physical transfer) from bilocation (consciousness projection), influencing visualization modes ([Danaan, 2025b](#danaan-2025b)).

## 5. Magic and Metaphysical Phenomena
Magic is an unexplained phenomenon connecting beings to the natural world, with implications for modeling consciousness and environmental interactions.

- **Definition**: Magic (white or dark) involves ancestral practices to influence reality, distinct from Crowley’s magick (dark occultism). White magic, as practiced by Druids, focuses on harmony and healing, potentially affecting the multiverse’s holographic structure.
- **Implementation**: The `Umgebung::Model::Galaxy` class includes a `magic_influence_` attribute (e.g., `float`) to model environmental effects. The `Umgebung::Physics::CUDAKernels` simulates magic as perturbations in the holographic field, inspired by Haramein’s vacuum fluctuations.
- **Sources**:
  - Danaan (2025) defines magic as a cultural term for unexplained phenomena, with white magic promoting harmony, influencing `Galaxy` attributes ([Danaan, 2025b](#danaan-2025b)).
  - Haramein (2013) models vacuum fluctuations as Planck oscillators, suggesting a mechanism for metaphysical perturbations in `CUDAKernels.cu` ([Haramein, 2013](#haramein-2013)).

## 6. Flower of Life and Holographic Reality
The Flower of Life (FoL) is a geometric pattern encoding the structure of reality, used to visualize the multiverse and gravitational fields.

- **Definition**: A sacred geometry pattern of 2D circles or 3D spheres forming a fractal lattice, visualized as a representation of Haramein’s PSU tiling. It encodes the multiverse’s structure and holographic gravitational fields, aligning with the concept of a unified field.
- **Implementation**: The `Umgebung::Model::Hierarchy` class uses a PSU-based voxel grid to define node connections, with `Umgebung::Physics::CUDAKernels` computing holographic interactions (e.g., gravity) based on Haramein’s surface-to-volume ratio φ. The `Umgebung::Render::View` visualizes this as a 3D FoL lattice or 2D FoL pattern.
- **Sources**:
  - Haramein (2013) uses PSU tiling to derive mass and gravity, with the surface-to-volume ratio φ linking quantum and cosmological scales. This provides the geometric basis for `Hierarchy` and CUDA simulations ([Haramein, 2013](#haramein-2013)).
  - Resonance Science Foundation (2024) describes the 3D Flower of Life as a visualization of PSU tiling, encoding gravitational fields for objects like protons or galaxies, applied to `Hierarchy`’s voxel grid and `View` rendering ([RSF, 2024](#rsf-2024)).
  - Danaan (2025) describes the Flower of Life as the lattice structure of reality, with 3D Merkaba fractals converging to a singularity, influencing the fractal design of `Hierarchy` ([Danaan, 2025a](#danaan-2025a)).

## References
- <a name="danaan-2025a"></a>Danaan, E. (2025). *The Flower of Life: The code of creation*. Retrieved from [https://www.elenadanaan.org/the-flower-of-life](https://www.elenadanaan.org/the-flower-of-life) (accessed July 1, 2025).
- <a name="danaan-2025b"></a>Danaan, E. (2025). *Debunk: Addressing misinformation*. Retrieved from [https://www.elenadanaan.org/debunk](https://www.elenadanaan.org/debunk) (accessed July 1, 2025).
- <a name="haramein-2013"></a>Haramein, N. (2013). Quantum Gravity and the Holographic Mass. *Physical Review & Research International, 3*(4), 270-292. Retrieved from http://www.sciencedomain.org/review-history.php?lid=2&id=4&aid=1298. Licensed under CC BY 3.0 (http://creativecommons.org/licenses/by/3.0).
- <a name="rsf-2024"></a>Resonance Science Foundation. (2024). Nassim Haramein on the Flower of Life and gravitational fields. *Facebook Reel*. Retrieved from https://www.facebook.com/reel/261504393354584 (accessed July 2, 2025).

## Notes
- Haramein’s 2013 paper provides the PSU tiling framework for spacetime quantization, visualized as a 3D Flower of Life lattice in Resonance Science Foundation (2024) for *Umgebung*’s voxel-based modeling and rendering.
- Additional sources (e.g., Alex Collier, Tony Rodrigues, Dan Willis, Chris Essonne, JP) will be included as their contributions are integrated.
- Citations link directly to `Umgebung`’s implementation, ensuring relevance to the C++/CUDA codebase.