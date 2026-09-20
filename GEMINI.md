# GEMINI.md: Umgebung Development & Architecture Guide

## 1. Project Overview & Core Philosophy

**Umgebung** is an interactive reality simulation engine designed to simulate quantum spatial dynamics modeled on Planck Spherical Units (PSUs) arranged in geometric structures such as the Flower of Life. The core engine natively calculates spatial parameters, including the fundamental PSU radius $r = \frac{\ell}{2}$ where $\ell$ is the Planck length.

The software is engineered with a **hardware-agnostic, decoupled architecture**. The core compute library (`libumgebung`) is completely separated from user interface clients, allowing the simulation to execute interchangeably across local development machines (Windows 11 / WSL2), headless SSH-only high-performance computing (HPC) clusters (UMaine ARCSIM, UNH Premise), or CPU-only supercomputers (RIT).

---

## 2. Target Hardware Topology Matrix

The application automatically scales memory budgets, thread pools, and CUDA block configurations at runtime to adapt to host hardware capabilities.

| Target System | CPU Specs | GPU / Accelerator Specs | Execution Strategy |
| --------------| --- | --- | --- |
| **Desktop PC** | AMD Ryzen 9 3900XT (12c/24t) | NVIDIA TITAN RTX (24 GB VRAM) | **Local CUDA Engine:** Large continuous grid allocations directly in GPU VRAM.

 |
| **Laptop PC** | Intel Core i9-13900H (20t) | NVIDIA RTX 4070 Laptop (8 GB VRAM) | **Local CUDA Engine:** Auto-tuned streamed tile chunking to prevent VRAM overflow.

 |
| **UMaine ARCSIM** | AMD EPYC Nodes | NVIDIA DGX A100 / RTX Multi-GPU | **Headless CUDA Engine:** Multi-GPU scaling via NVLink and SLURM batch jobs.

 |
| **UNH Premise** | AMD EPYC / Intel Xeon | NVIDIA A100 / V100 GPUs | **Headless CUDA Engine:** Automated VRAM discovery and stream graph1. **Scaffold the Directory Architecture** |
| From the root of your empty `.git` repository (e.g., `C:\dev\Umgebung`), generate the specialized subdirectories that will house your codebase. Create a `src/` directory containing `core/`, `cli/`, and `gui/` modules, alongside a dedicated `include/umgebung/` path for your public engine headers. Create additional top-level directories for `tests/` and `packaging/` to isolate your functional testing and deployment logic from the application source.

 |  |  |  |

2. **Embed the vcpkg Submodule**
To ensure the build system is portable across your local Windows machines and the Linux HPC clusters, embed the vcpkg package manager directly into the project rather than relying on a global system installation.



* Execute `git submodule add [https://github.com/microsoft/vcpkg.git](https://github.com/microsoft/vcpkg.git) submodules/vcpkg` to clone the toolchain repository.


* Navigate into the new `submodules/vcpkg` folder and run the provided bootstrap script (`bootstrap-vcpkg.bat` on Windows or `./bootstrap-vcpkg.sh` on WSL2/Linux) to compile the vcpkg executable.



3. **Configure the vcpkg Manifest & Presets**
Return to the repository root and declare your project's dependencies by creating a `vcpkg.json` manifest file. You can initialize this manually or run `vcpkg new --application` followed by `vcpkg add port fmt` to establish a baseline dependency graph.
Next, create a `CMakePresets.json` file to explicitly route CMake through the vcpkg toolchain.



```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "default",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/out/build/x64-debug",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake"
      }
    }
  ]
}

```

4. **Establish the Root CMakeLists.txt**
Create the primary build script (`CMakeLists.txt`) at the repository root. This configuration must require a minimum CMake version of 3.21, enforce C++20 functionality, and pull in your newly created subdirectories.



```cmake
cmake_minimum_required(VERSION 3.21)
project(Umgebung LANGUAGES CXX CUDA)

enable_testing()

# Project wide setup
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Disable MSBuild file tracking if using caching tools in Visual Studio[cite: 3]
list(APPEND CMAKE_VS_GLOBALS TrackFileAccess=false)

# Main targets built by this project
add_subdirectory(src)

# Wrap auxiliary directories to prevent execution if built as a subproject[cite: 3]
if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
    add_subdirectory(packaging)
endif()

```

5. **Isolate Development Environments via .gitignore**
Create a `.gitignore` file at the root to prevent your repository from tracking compilation artifacts and localized IDE settings. You should explicitly exclude the `out/` build directory, Visual Studio's `.vs/` cache databases (which store `ENGINE.ipch` and Copilot indexes), and any `CMakeUserPresets.json` files.


6. **Link Component Targets in Source**
Inside the `src/` directory, create a secondary `CMakeLists.txt` file. This localized build script will logically group and compile the simulation's components by defining the `umgebung_core` library (reading from `core/`), the headless `umgebung-cli` executable, and the interactive `umgebung-gui` interface.



Which of the three source targets (`core`, `cli`, or `gui`) would you like to populate with foundational C++ code first?