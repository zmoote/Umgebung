# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the "Umgebung" project.

## Project Overview

"Umgebung" is a C++ and CUDA-based 3D rendering application. Its primary goal is to create a visual representation of reality based on concepts from the UFO Disclosure Community. The project is in its early stages of development.

The application uses an Entity-Component-System (ECS) architecture, leveraging the `EnTT` library for entity management. The rendering pipeline is built on OpenGL, using `glad` for loading OpenGL functions and `glfw` for window and input management. The user interface is built with `Dear ImGui`, featuring several custom panels for interacting with the scene, such as a hierarchy viewer, properties editor, and a viewport.

Key technologies and libraries used:
- **C++17**: The primary programming language.
- **CUDA**: For potential GPU-accelerated computations.
- **CMake**: The build system used for the project.
- **vcpkg**: For managing third-party dependencies.
- **OpenGL**: The graphics API for rendering.
- **EnTT**: A header-only, dependency-free, and C++17-compliant entity-component-system (ECS) library.
- **glm**: A header-only C++ mathematics library for graphics software.
- **imgui**: A bloat-free graphical user interface library for C++.
- **spdlog**: A fast, header-only/compiled, C++ logging library.
- **nlohmann/json**: A JSON library for modern C++.
- **assimp**: A library to import and export various 3d-model-formats.
- **PhysX**: For physics simulation.

## Building and Running

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

3.  **Run the application:**
    The executable will be located in the `out/build/<preset-name>/bin` directory.
    ```bash
    # For a debug build
    ./out/build/x64-debug/bin/Umgebung.exe

    # For a release build
    ./out/build/x64-release/bin/Umgebung.exe
    ```

## Development Conventions

- The codebase is organized into namespaces (e.g., `Umgebung::app`, `Umgebung::renderer`).
- Header files use `#pragma once` for include guards.
- Modern C++ features, including smart pointers (`std::unique_ptr`, `std::shared_ptr`), are used for memory management.
- Private and protected class member variables are suffixed with an underscore (e.g., `window_`).
- The project has a custom logging utility (`Umgebung::util::Logger`) that should be used for logging messages.

## Research Submodule

The `submodules/research` directory contains a collection of documents, papers, and personal notes that provide the foundational knowledge and inspiration for the "Umgebung" project. The contents are organized into the following subdirectories:

### `Computation`

This directory contains technical literature related to computer graphics, programming, and simulation. The materials cover:

- **CUDA Programming**: Guides and documentation for programming with NVIDIA's CUDA platform.
- **CMake**: Best practices and guides for using the CMake build system.
- **Game Engine Architecture**: Books and papers on the design and implementation of game engines.
- **Real-time Rendering**: Resources on the techniques and algorithms for real-time graphics rendering.
- **General Relativity**: A paper on a CUDA-based ray-tracer in general relativity.

### `Other`

This directory contains a mix of scientific textbooks and esoteric materials, including:

- **Physics and Astronomy**: Standard university-level physics and astronomy textbooks.
- **Esoteric and UFO-related Documents**: Materials on topics such as crystals and UFO contact, which align with the project's goal of exploring alternative views of reality.

### `Personal`

This directory contains personal notes and documents related to the project's development and the developer's setup. Key files include:

- **`Zach wants to create an interactive.txt`**: A detailed document outlining the project's vision, goals, and the philosophical underpinnings of the simulation. It explicitly states the desire to model reality based on information from "fringe" thinkers and extraterrestrial contactees, covering topics like Consciousness, Soul, and Vibrational Density alongside Quantum and Classical Mechanics.
- **Hardware Specifications**: Text files detailing the specifications of the developer's custom-built PC and laptop.
- **`Potential Classes For Umgebung.txt`**: A list of potential classes for the project, such as `Camera`, `Shader`, `Model`, and `Mesh`.

### `Thinkers`

This directory contains folders named after individuals who are influential to the project's philosophy. These individuals are mentioned in the project's `README.md` and the `Zach wants to create an interactive.txt` file as sources of inspiration. The list of thinkers includes:

- Alex Collier
- Billy Carson
- Chris Essonne
- Dan Willis
- Dani Henderson
- Elena Danaan
- Nassim Haramein
- Randall Carlson
- Sacha Stone
- Tom Campbell
