# Umgebung
Umgebung is an open-source C++ project with the goal of accurately representing reality by understanding and implementing concepts and knowledge shared by individuals in the UFO Disclosure Community including but not limited to:
- Alex Collier
- Elena Danaan
- Dan Willis
- Tony Rodrigues
- Chris Essone
- Nassim Haramein
- JP (US Army Insider)


This is a personal project, open-source on GitHub, and not expected to have wide use but is available for those who may be interested. Built with C++ and CUDA, using vcpkg for dependencies and CMake/Ninja for builds. More dependencies will be added or changed in the future.

## Features

- **2D and 3D Views**: Seamlessly switch between a 2D hierarchical view for navigating the multiverse and a 3D view for exploring specific universes, galaxies, and other celestial bodies.
- **Entity-Component-System (ECS) Architecture**: Leverages the EnTT library for flexible and scalable entity management.
- **OpenGL Rendering**: Utilizes OpenGL for high-performance 3D graphics rendering.
- **Dear ImGui Interface**: Provides an interactive user interface with custom panels for scene interaction, including a hierarchy viewer, properties editor, and a viewport.
- **Scene Serialization**: Supports saving and loading of scene data to and from JSON files.

## Prerequisites
- OS: Windows 10/11 (Windows is the priority for now, but Linux support may be added in the future.)
- GPU: NVIDIA Graphics Card with CUDA support (Targeting the 'Turing' and 'Ada Lovelace' CUDA architectures).
  - *For reference, the developer's machines include:*
    - *Laptop: NVIDIA RTX 4070 Laptop GPU, Intel Core i9-13900H CPU, 32 GB DDR5 RAM.*
    - *Desktop: NVIDIA TITAN RTX GPU, AMD Ryzen 9 3900XT CPU, 32 GB DDR4 RAM.*
- RAM and VRAM requirements are currently unknown, but development is done on systems with 32 GB RAM. 

## Installation and Building

The project is set up to be built on Windows using CMake and the Ninja build system.

### Prerequisites

- Windows 10/11
- CMake
- Ninja
- A C++ compiler (e.g., MSVC from Visual Studio)
- An NVIDIA GPU with CUDA support (for CUDA features)

### Build Steps

1.  **Configure the project using a CMake preset:**
    ```bash
    # For a debug build
    cmake --preset x64-debug

    # For a release build
    cmake --preset x64-release
    ```

2.  **Build the project:**
    ```bash
    # For a debug build
    cmake --build out/build/x64-debug

    # For a release build
    cmake --build out/build/x64-release
    ```

## Usage

The executable will be located in the `out/build/<preset-name>/bin` directory.
```bash
# For a debug build
./out/build/x64-debug/bin/Umgebung.exe

# For a release build
./out/build/x64-release/bin/Umgebung.exe
```

## Project Structure

- `assets/`: Contains configuration files, icons, 3D models, shaders, and textures.
- `docs/`: Project documentation.
- `include/`: Header files for the project, organized by module (app, ecs, renderer, scene, ui, util).
- `src/`: Implementation files corresponding to the headers in `include/`.
- `submodules/`: External repositories, including `research` (for foundational knowledge) and `vcpkg` (for dependency management).

## License
Umgebung is licensed under Apache 2.0 -- see LICENSE.md for details.

## Acknowledgements
Inspired by the works and experiences of Alex Collier, Elena Danaan, Dan Willis, Tony Rodrigues, Chris Essone, and others.
Acknowledgements and references used will be updated over time.

- https://www.elenadanaan.org/
- https://thewebmatrix.net/
- https://www.tonyrodrigues.com/
- https://www.gsjournal.net/Science-Journals-Papers/Author/2334/Chris,%20Essonne

## Dependencies

This project uses vcpkg to manage the following third-party dependencies:
- assimp
- cuda
- eigen3
- entt
- glad (gl-api-latest, loader, wgl)
- glfw3
- glm
- imgui (docking-experimental, glfw-binding, opengl3-binding)
- nlohmann-json
- physx
- spdlog
- stb

## Contact
For questions or feedback, reach out via GitHub issues or nuluumo@gmail.com.