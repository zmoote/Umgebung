# Umgebung

**Umgebung** is a GPU-accelerated, multi-scale 2D/3D reality simulation engine and computational physics testbed built with **C++20**, **CUDA**, **CMake**, and **vcpkg**, paired with an optional **Lean 4** formal verification layer.

## Vision & Three-Pillar Architecture

1. **Mainstream Physics Baselines (`models/mainstream`):**
   - Verified standard physics solvers (e.g., Newtonian/Post-Newtonian N-body gravity, symplectic integrators, Maxwell electrodynamics) used for analytical verification, conservation-law testing, HPC scaling benchmarks, and controlled A/B comparisons.
2. **Unified Research Models (`models/unified`):**
   - Computable, verbatim implementations of unified physics, cosmology, spacetime geometry, and consciousness models synthesized from the `submodules/research` corpus.
3. **Formal Mathematical Rigor (`formal/`):**
   - An optional **Lean 4** specification and proof layer that formalizes axioms, physical dimensions, conservation theorems, and discretization properties independently of the GPU runtime.

Each universe is stored in its own `.umgebung` scene file specifying its hierarchy, initial conditions, and active physics profile, allowing direct quantitative comparison between mainstream and unified models under identical initial states.

## Target Platforms & Execution Modes

Umgebung builds from a single codebase into two modes:
- **Desktop Interactive (`UMGEBUNG_BUILD_GUI=ON`):** Windows & Linux interactive GUI with GLFW, Vulkan/NVRHI, and Dear ImGui.
- **Headless HPC (`UMGEBUNG_BUILD_GUI=OFF`):** Dependency-light CLI/batch executable designed for Linux NVIDIA GPU clusters (V100, A100, RTX/TITAN nodes).

## Quick Start

```bash
# Clone with submodules
git clone --recursive <repo-url>
cd Umgebung

# Configure & build (Linux Desktop GUI)
cmake --preset linux-desktop-debug
cmake --build --preset linux-desktop-debug

# Configure & build (Headless HPC)
cmake --preset hpc-headless-release
cmake --build --preset hpc-headless-release
```

See [GEMINI.md](GEMINI.md) for the full architecture guide, hardware topology matrix, and repository layout, and [formal/README.md](formal/README.md) for Lean 4 formal verification details.
