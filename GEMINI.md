# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the "Umgebung" project.

## Project Overview

"Umgebung" is a C++ and CUDA-based 3D rendering application. Its primary goal is to create a visual representation of reality based on concepts from the UFO Disclosure Community. This project serves as the Senior Capstone for the user's Physics B.S. program at SNHU (expected graduation Spring 2026). The simulation aims to span the entire scale of reality, from the Multiverse down to the most fundamental unit. The project is in its early stages of development.

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

**Note: The user prefers to run CMake configuration and build commands manually via Visual Studio. The assistant should not execute `cmake` commands in the CLI.**

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

## Development Status (As of March 2026)

### Optimization & Scalar Field Integration (March 2026)
The simulation has been optimized for high-entity counts and expanded with esoteric energy mechanics.

*   **Transform Matrix Caching**: 
    *   Added a `dirty` flag and `cachedModelMatrix` to the `Transform` component.
    *   The model matrix is only recalculated if `position`, `rotation`, or `scale` changes, significantly reducing CPU overhead for static objects like the Multiverse lattice.

*   **RenderSystem Batching Optimization**:
    *   The `RenderSystem` now reuses `InstanceData` vectors across frames, minimizing memory allocations.
    *   Optimized the instancing loop to leverage the cached `Transform` matrices.

*   **Phryll & Scalar Field (Observer Effect)**:
    *   **PhryllComponent**: Added a component to model "Life Force" energy density and vibrational frequency.
    *   **ScalarFieldSystem**: Implements the **Observer Effect**. Entities in the camera's focus (viewing frustum and proximity) experience an increase in `phryllInfluence`.
    *   **Vibrational Manifestation**: Higher-density entities (5th density+) now only "manifest" (become visible) when the observer's focus (Phryll density) is high enough.
    *   **Shader Integration**: Shaders now implement a "Phryll Glow" effect and dynamic transparency based on manifestation state.

### Multi-Scale CUDA Physics (Updated)
... (rest of the file)
*   **Multiverse Lattice (3D Flower of Life)**: 
    *   Implemented a `MultiverseSystem` that procedurally generates "Bubble Universes" in an **Interconnected Hexagonal Close Packing (HCP)** lattice.
    *   In this geometry, the radius of each universe sphere equals the center-to-center spacing, ensuring that each bubble's center sits on the surface of its neighbors to form the 3D Flower of Life blueprint.
    *   A new **"Genesis"** menu in the UI allows users to trigger this generation with configurable layers and spacing.

*   **Time and Density Mechanics**:
    *   **TimeComponent**: Every entity now possesses a `density` value (1.0 to 13.0), representing its vibrational frequency.
    *   **Subjective Time Flow**: The `PhysicsSystem` calculates a per-entity `subjectiveDt`. Higher density entities experience faster subjective time compared to physical (3rd density) matter.
    *   **Gravity-Time Entanglement**: Proximity to `Planetary` scale bodies creates a linear time matrix. Entities can be toggled as `isTargetedByGravity = false`, placing them in "The Void"—a non-temporal state where subjective time is zero.

*   **Source View (3-6-9 Lattice Visualization)**:
    *   Added a toggle in the **Statistics Panel** to enable "Source View".
    *   When active, shaders transition from physical rendering to a geometric "Source Code" representation. Meshes reveal a glowing **3-6-9 grid**, and galactic/particle points transform into **9-pointed fractal stars**.
    *   Shaders implement vibrational pulsing and Fresnel edge-glow, with the pulse frequency driven by the entity's `density`.

### Multi-Scale CUDA Physics
*   **Per-Particle Delta-Times**: The `MicroPhysics.cu` kernel has been upgraded to accept an array of delta-times. This allows the GPU to physically simulate relative time flow, where different groups of particles in the same simulation can experience time at different rates.
*   **Initialization Fix**: Resolved a critical race condition where CUDA-GL buffer registration occurred before the driver context was fully initialized by the Physics System.

### Navigation and UI Enhancements
*   **Extreme Scale Observation**:
    *   Unified the camera unit system in `CameraLevels.json` to absolute kilometers.
    *   Extended the far clipping plane to **1e30 units** to prevent visual clipping at the Universal and Multiversal scales.
    *   Adjusted LoD logic to ensure "Bubble Universes" render as 3D meshes rather than dots at the Universal scale.
*   **Dynamic Navigation**:
    *   Camera movement speed now scales automatically based on the current `ObserverScale`. 
    *   Navigation at the Universal scale is quintillions of times faster than at the Human scale.
    *   Implemented a **Sprint Modifier (Left Shift)** for a 10x speed boost.
*   **Refined Logging**: Implemented conditional "State Change" logging for systems to reduce console noise while maintaining visibility into significant events (e.g., scale transitions, gravity source detection).

### PhysX Integration (Legacy)
*   A `PhysicsSystem` manages PhysX worlds across multiple scales using `ScaleComponent` and `SimScale` normalization.
*   Supports `RigidBody` and `Collider` components (Box, Sphere, ConvexMesh) with runtime updates and GPU acceleration via `PxCudaContextManager`.

## Research Submodule

The `submodules/research` directory contains technical and esoteric foundational materials for the project, organized by subject (Computation, Personal, Other) and Thinker (e.g., Alex Collier, Nassim Haramein, Elena Danaan).

---
*Note: The project status is updated regularly to reflect the integration of new theoretical physics and consciousness simulation features.*