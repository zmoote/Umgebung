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