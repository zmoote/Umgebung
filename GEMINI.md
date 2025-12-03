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

## Development Status (As of November 2025)

### PhysX Integration
The NVIDIA PhysX engine has been integrated into the project to handle physics simulations.
- A `PhysicsSystem` (`src/ecs/systems/PhysicsSystem.cpp`) was created to manage the PhysX world. It is initialized and updated in the main `Application` class.
- A `RigidBody` component (`include/umgebung/ecs/components/RigidBody.hpp`) was added to allow entities to participate in the physics simulation.
- A `Collider` component (`include/umgebung/ecs/components/Collider.hpp`) was added to define the physical shape of an entity for collision detection.
- The `PhysicsSystem` creates a `PxRigidActor` for each entity with a `RigidBody` and `Transform` component. It now requires a `Collider` component to be present to create and attach a `PxShape` to the actor. This resolves the issue of physics objects falling through each other. Supported collider types are `Box`, `Sphere`, and `ConvexMesh`. The `ConvexMesh` type uses the PhysX cooking library to generate a collider from the entity's visual mesh for more accurate collisions.
- The `PhysicsSystem` now correctly handles runtime changes to `RigidBody` and `Collider` properties. When a component is modified in the UI, it is flagged as 'dirty', prompting the system to remove the old `PxRigidActor` and create a new one with the updated properties. This fixes issues with changing an object from static to dynamic and ensures transform changes are respected.
- The `SceneSerializer` has been updated to correctly save and load entities with `RigidBody` and `Collider` components.

### Scene Management
- The application now supports saving and loading scenes to and from different files.
- The "File" menu now includes "Save Scene", "Save As...", and "Open Scene..." options.
- A file picker is used to select the file path for saving and loading scenes. When saving, the file picker now automatically appends the `.umgebung` extension if one is not provided.
- The application automatically loads a `default.umgebung` scene on startup from the `assets/scenes/` directory.
- Scene files are now organized in `assets/scenes/`.

### Simulation Mode
- Added "Simulation" menu with **Play**, **Stop**, and **Pause** controls.
- **Editor State**: The default state. Physics simulation is paused, and the scene can be edited.
- **Simulate State**: Physics simulation runs.
    - **Play**: Saves the current scene state to a temporary file (`assets/scenes/temp.umgebung`) and starts the physics simulation.
    - **Stop**: Stops the simulation, resets the physics system, and reloads the scene from the temporary file, restoring the initial state.
    - **Pause**: Toggles the physics simulation update without resetting the scene.

### Multi-Scale Physics Implementation
- **Architecture**: The `PhysicsSystem` uses a **Single-Physics, Multi-Scene architecture**. A single `PxFoundation` and `PxPhysics` instance are shared across the entire application to adhere to PhysX singletons constraints.
- **SimScale**: A `simScale` factor is calculated for each `ScaleType`. This factor normalizes ECS units (which represent vast distances like Light Years) into PhysX units (approx. 1.0 unit = \"typical object size\" at that scale).
    - This ensures that the physics engine always operates within its optimal floating-point range (0.1 to 100.0 units) regardless of whether the object is a proton or a galaxy.
    - Gravity and other forces are scaled accordingly (`Gravity = -9.81 * simScale`).
- **ScaleComponent**: A new `ScaleComponent` (`include/umgebung/ecs/components/ScaleComponent.hpp`) defines the scale of an entity.
- **Multi-Scene Management**: The `PhysicsSystem` maintains a map of `ScaleType` to `PhysicsWorld` structs. Each `PhysicsWorld` contains a `PxScene` and the scale-specific `simScale` factor.
    - The `PhysicsSystem` automatically places an entity's PhysX actor into the correct `PxScene` based on its `ScaleComponent`.
    - Changing an entity's scale at runtime automatically migrates its physics actor to the new scene.
- **Serialization**: `ScaleComponent` is fully serializable, allowing scale data to be saved and loaded with scenes.
- **UI**: The Properties Panel now includes a "Scale" section to view and edit an entity's `ScaleType`.

### GPU Acceleration Status (Fixed)
The effort to enable GPU-accelerated physics via CUDA (`PxSceneFlag::eENABLE_GPU_DYNAMICS`) was initially paused due to a runtime exception (`0xC0000005: Access violation`) that occurred only in debug builds. The issue has been resolved, and GPU acceleration is now functional.

- **Problem**: Enabling GPU dynamics caused a crash during scene creation (`gPhysics_->createScene()`) in debug builds. The root cause was a configuration issue in the build system that led to a mismatch between the debug-compiled application and release-compiled PhysX libraries. This also resulted in the debugger being unable to find PDB files for the PhysX GPU libraries.
- **Diagnosis**: The investigation revealed that the `vcpkg` port for `unofficial-omniverse-physx-sdk` was not correctly differentiating between debug and release library versions. Furthermore, several required DLLs and libraries for the GPU simulation and visual debugger were not being correctly linked or copied to the build output directory.
- **Solution**: The `CMakeLists.txt` file was modified to manually manage the PhysX dependency. This involved:
    1. Removing the `find_package` call for `unofficial-omniverse-physx-sdk`.
    2. Manually specifying the paths to the correct `debug` and `release` PhysX libraries.
    3. Adding the `PhysXPvdSDK_static_64` library to resolve linker errors.
    4. Adding custom commands to copy the required `PhysXGpu_64.dll` and `PhysXDevice64.dll` to the output directory for both debug and release builds.
- **Current State**: The application now correctly builds and runs with PhysX GPU acceleration enabled in both debug and release configurations. The fallback to CPU physics also functions correctly if a capable GPU is not found.

### UI/UX Updates
- The `RigidBody` component can now be added via the Properties Panel.
  - The "Dynamic" option in the "Add Component" menu now correctly adds a `RigidBody` component with its `BodyType` set to `Dynamic`.
  - The Properties Panel now displays and allows editing of the `mass` and `BodyType` properties of an existing `RigidBody` component.
- The `Collider` component can now be added and edited via the Properties Panel.
  - A "Collider" option has been added to the "Add Component" menu.
  - The Properties Panel now displays and allows editing of the `Collider`'s properties, such as shape type (`Box`, `Sphere`, `ConvexMesh`) and dimensions (half-extents or radius).
- A `DebugRenderSystem` has been added to visualize physics colliders.
  - A "Show Physics Colliders" checkbox is now available in the "Tools" -> "Statistics" panel.
  - When enabled, static colliders are drawn in green and dynamic colliders in red.
- **Logging**: Log files are now saved in a dedicated `logs/` directory. The `Logger` class automatically creates this directory if it doesn't exist.

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

## Codebase Analysis

### `assets` directory:
*   **`config/CameraLevels.json`**: Defines different camera settings (near/far planes, units) for various scales of the simulation (Planetary, SolarSystem, Galactic, etc.). This is a crucial file for controlling the camera's behavior at different zoom levels.
*   **`icon`**: Contains the application icon in different formats.
*   **`models`**: Contains `.glb` files for basic geometric shapes (Cube, Sphere, etc.). These are likely used for placeholder or simple representations of objects in the scene.
*   **`shaders`**: Contains GLSL shader files (`.vert`, `.frag`). The current shaders are very simple, with a vertex shader for transforming vertices and a fragment shader for outputting a uniform color.
*   **`textures`**: Currently empty, but intended to hold textures for models.

### `include` directory:
*   This directory contains all the header files (`.hpp`) for the project, organized into subdirectories that mirror the `src` directory structure.
*   **`umgebung/app`**: `Application.hpp` defines the main application class, which manages the main loop, window, renderer, scene, and UI.
*   **`umgebung/asset`**: `ModelLoader.hpp` declares the class responsible for loading 3D models using Assimp. It includes a cache to avoid reloading models.
*   **`umgebung/ecs`**: This is the core of the Entity-Component-System architecture.
    *   **`components`**: Defines various components that can be attached to entities, such as `Transform`, `Renderable`, `Name`, `Soul`, and `Consciousness`. The `Soul` and `Consciousness` components are currently empty placeholders, reflecting the project's unique goals. The components are set up for serialization with `nlohmann/json`.
    *   **`entities`**: Contains classes that seem to represent hierarchical concepts (Multiverse, Universe, Galaxy, etc.), but they are not directly used as ECS entities. They seem to be more like conceptual data structures. The actual entities are created in the `Scene` class.
    *   **`systems`**: `RenderSystem.hpp` declares the system responsible for rendering entities that have both a `Transform` and a `Renderable` component.
*   **`umgebung/renderer`**:
    *   `Camera.hpp`: A class for managing the camera's position, orientation, and projection.
    *   `Framebuffer.hpp`: A class for creating and managing an OpenGL framebuffer, which is used for rendering the scene to a texture.
    *   `Mesh.hpp`: Represents a 3D mesh with vertices and indices, and handles the OpenGL vertex array and buffer objects.
    *   `Renderer.hpp`: The main rendering class that manages the shader, camera, and model loader.
    *   `gl/Shader.hpp`: A wrapper for an OpenGL shader program, which handles loading, compiling, and setting uniforms.
*   **`umgebung/scene`**:
    *   `Scene.hpp`: Manages the `entt::registry` for the ECS, and handles entity creation and destruction.
    *   `SceneSerializer.hpp`: A class for serializing and deserializing the scene to and from a JSON file.
*   **`umgebung/ui`**:
    *   `UIManager.hpp`: Manages the ImGui user interface, including the dockspace and all the panels.
    *   `Window.hpp`: A wrapper for the GLFW window.
    *   `imgui`: Contains the individual UI panels, such as the `HierarchyPanel`, `PropertiesPanel`, `ViewportPanel`, `ConsolePanel`, etc.
*   **`umgebung/util`**:
    *   `Logger.hpp`: A singleton logger class that uses `spdlog` to provide logging to the console, a file, and the ImGui console panel.
    *   `LogMacros.hpp`: Defines macros for easy logging.
    *   `JsonHelpers.hpp`: Provides `nlohmann/json` serializers for `glm` types (`vec3`, `vec4`, `quat`).

### `src` directory:
*   This directory contains the implementation files (`.cpp`) for the classes declared in the `include` directory.
*   **`Main.cpp`**: The entry point of the application. It initializes the logger and the `Application` class.
*   The rest of the `.cpp` files provide the implementation for the classes in the corresponding header files.

### Overall Architecture:
The project follows a modern C++ ECS architecture. The use of `EnTT` for the ECS, `glm` for math, `glad`/`glfw` for OpenGL, and `ImGui` for the UI is a standard and effective combination for this type of application. The code is well-organized into namespaces and subdirectories. The serialization of the scene to JSON is a key feature, allowing for saving and loading scene data. The project's unique aspect is the inclusion of components like `Soul` and `Consciousness`, which are currently placeholders but indicate the project's philosophical direction.

## Future Development / Multi-Scale Physics (TODO) 

Achieving the goal of simulating all scales in a single application requires a coupled architecture.

### 1. Cross-Scale Coupling and CUDA Integration

* **Inter-Scene Force/Effect Coupling:** Develop custom code (likely within the `PhysicsSystem::Update` loop) to manage physics interaction *between* the scenes.
    * **Gravity Transfer:** Calculate the total force (e.g., gravitational pull) from the Macro Scene entities and apply it as the gravity vector in the Meso Scene.
    * **Shifting Origin:** Implement `PxScene::shiftOrigin()` based on the camera's position or the primary focus entity's position to maintain precision in the Macro and Meso scenes.
* **CUDA Micro-Scale Solver:** Integrate a **CUDA kernel** to handle the physics for Micro-scale particles (e.g., fluid dynamics, molecular interaction).
    * **Bypass PhysX:** These entities would use the CUDA solver instead of being added to a `PxScene`.
    * **Force Accumulation:** Implement the kernel to calculate the **net force and torque** exerted by a cloud of these CUDA-managed particles onto any **Meso-scale** rigid body they intersect, applying the result via `PxRigidBody::addForce()`.

### 2. Rendering Considerations

* **Scale-Dependent LoDs (Levels of Detail):** Integrate the `ScaleComponent` with the `RenderSystem` to switch rendering methods based on scale (e.g., use point sprites for distant Macro objects; full meshes for Meso objects).
* **Camera Integration:** Use the camera's current zoom/position (and the data in `assets/config/CameraLevels.json`) to control which scale of physics is currently being observed and, potentially, prioritize updates for the visible scale.
