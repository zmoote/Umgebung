# GEMINI.md: Umgebung Development & Architecture Guide

## 1. Project Overview & Core Philosophy

**Umgebung** is an interactive 2D/3D multi-scale reality simulation engine and computational physics testbed written in **C++20** and **CUDA**. It is designed to operate across three tightly coupled pillars:

1. **Mainstream Physics Baselines (`models/mainstream/`):**
   - Verified, textbook physics solvers (e.g., Newtonian and Post-Newtonian N-body gravity, NFW/Dark Matter halos, symplectic integrators, Maxwell electrodynamics, classical/quantum reference systems).
   - Serves as an academically rigorous verification suite (analytical benchmarks, conservation-law checks, numerical convergence, and CUDA/HPC scaling) and as a controlled baseline for side-by-side comparison against alternative theories.
2. **Individual Thinker Modules (`models/thinkers/`) & Unified Synthesis (`models/unified/`):**
   - Implements concepts and mathematics **verbatim** (not merely as loose inspiration) from the `Thinkers` in the `submodules/research` submodule, including but not limited to:
     - Elena Danaan
     - Alex Collier
     - Nassim Haramein
     - Chris Essonne
     - Christopher Cooper
     - Dan Willis
     - Marcel Vogel
     - Salvatore Pais
     - Masaru Emoto
   - **Per-Thinker Isolation (`models/thinkers/`):** Each Thinker is implemented as a self-contained, independently runnable and verifiable module (e.g., `Essonne`, `Haramein`, `Pais`, `Danaan`, `VogelWillis`, `Collier`, `Emoto`, `Cooper`) so individual theories can be tested cleanly on their own without cross-contamination.
   - **Unified Synthesis (`models/unified/`):** Composes all individual Thinker modules together into one singular, unified multi-scale reality model spanning gravity & cosmology, electromagnetism & devices (Pais HEEMFG, Vogel/Eredyon Phryll crystal), consciousness/metaphysics/geometry (Flower of Life, the Void, Love, non-linear time, vibrational density), and multi-scale zoom (Multiverse &rarr; Bubble Universe &rarr; Galaxy &rarr; Star &rarr; Planet &rarr; Entity &rarr; Atom &rarr; Subatomic PSU).
3. **Formal Mathematical Verification (`formal/` — Lean 4):**
   - Uses **Lean 4** (with Lake) as an offline formal specification, proof, and reference-oracle layer alongside C++/CUDA.
   - Explicitly separates **proven consequences** from **postulated axioms**, verifies dimensional consistency and algebraic/discretization properties (e.g., symplecticity, time-reversibility, and discrete conservation), and integrates optionally with CTest without introducing any runtime or build dependency on HPC clusters.

---

## 2. Target Hardware Topology Matrix

The application automatically fetches system configurations at runtime to adapt to host hardware capabilities.
Here I list my two personal machines, followed by the High-Performance Computing (HPC) resources at various universities, which I plan to apply to for a Doctorate Physics or Astrophysics Ph.D. Program.
As of October 3rd, 2026, I am waiting to hear back from UMaine about their decision for Spring 2027; The other universities would be Fall 2027.

Systems will require an NVIDIA graphics card for CUDA acceleration.

| Target System | CPU | GPU | RAM | CUDA SM |
| --------------| --- | --- | --- | ------- |
| **Personal Desktop** | AMD Ryzen 9 3900XT (12c/24t) | NVIDIA TITAN RTX (24 GB VRAM) | 32 GB DDR4 | `75` (Turing) |
| **Personal Laptop** | Intel Core i9-13900H (14c (6 Performance cores, 8 Efficient cores)/20t) | NVIDIA RTX 4070 Laptop (8 GB VRAM) | 32 GB DDR5 | `89` (Ada Lovelace) |
| **UMaine ARCSIM** | AMD EPYC Nodes | NVIDIA DGX A100 / RTX Multi-GPU | &ge; 512GB | `80` / `75` / `89` |
| **UNH Premise** | AMD EPYC / Intel Xeon | NVIDIA A100 / V100 GPUs | &ge; 128GB | `80` / `70` (Volta) |
| **RIT SPORC** | Intel (x86_64) | 100x NVIDIA A100 GPUs | 24TB | `80` (Ampere) |

---

## 3. Architectural Principles & Execution Modes

- **Dual Execution Modes (Single Codebase):**
  - **Desktop Interactive (`UMGEBUNG_BUILD_GUI=ON`):** Full 2D/3D interactive viewport and context-aware hierarchy inspector using GLFW, Vulkan + NVRHI, and Dear ImGui.
  - **Headless HPC Batch (`UMGEBUNG_BUILD_GUI=OFF`):** Zero windowing/graphics dependencies; driven via CLI (`cxxopts`) and `.umgebung` files for batch sweeps, A/B comparisons, checkpoints, and multi-GPU cluster runs.
- **Three Explicit Simulation Scopes (One Shared Core Entity/Field Schema):**
  1. **Scenario (`scenes/scenarios/*.scenario.umgebung`):** Standalone computational physics experiments, device tests, and analytical benchmarks (e.g., Kepler 2-body, Figure-8 3-body, Spiral Galaxy Periphery, Pais HEEMFG cavity, Vogel/Eredyon Phryll crystal). Displayed cleanly as `Scenario → Entities / Devices / Fields` without forcing a Multiverse wrapper.
  2. **Bubble Universe (`scenes/universes/*.umgebung`):** Full multi-scale universe bubble supporting nested cosmological-to-subatomic hierarchy (`Universe → Galaxy → Star System → Planet → Conscious Entity / Crystal → Atom → Subatomic PSU`).
  3. **Flower of Life Multiverse (`scenes/multiverses/*.multiverse.umgebung`):** Specific to **Elena Danaan's framework & Unified Synthesis** — models the geometric **Flower of Life** lattice where each intersecting sphere/node references a `.umgebung` **Bubble Universe** bridged by **the Void**.
- **Composable Model Registry & Live Simultaneous A/B Overlay:**
  - Any Scenario defines initial conditions **once** and can be executed under any compatible model (`Mainstream::*`, standalone `Thinker::*`, or `Unified::Synthesis`) via live dropdown or CLI (`--model`).
  - Supports **simultaneous overlaid A/B comparison** (`--compare-model` / GUI Overlay toggle): runs Model A (primary) and Model B (comparison) side-by-side on the GPU from identical initial conditions, overlaying both trajectories and live quantitative divergence telemetry ($\Delta r, \Delta v, \Delta E, \Delta\tau$).
- **Separation of Formal Proofs and Compute:**
  - Lean 4 never runs inside GPU kernels or HPC batch jobs. Proof verification (`lake build`) is optional in CMake (`UMGEBUNG_VERIFY_PROOFS`, `OFF` by default) so headless HPC builds remain self-contained.

---

## 4. Repository Layout

```text
Umgebung/
├── assets/                     # Icons, shaders, and runtime resources
├── formal/                     # Lean 4 formal specifications & proofs (optional)
│   └── Umgebung/
│       ├── Algebra/            # Self-contained commutative field & Vec3 algebra
│       ├── Mainstream/         # Standard physics axioms, conservation laws, symmetries
│       ├── Unified/            # Formalized Thinkers models & explicit axioms
│       └── Numerical/          # Discretization, integrator symplecticity & stability proofs
├── include/umgebung/           # Public C++20 / CUDA headers (mirrored by src/)
│   ├── core/                   # Application, SystemProfiler, logging, CLI, timing
│   ├── compute/                # CUDA context, streams, DeviceBuffer, VRAM budget tracker
│   ├── scene/                  # Shared Entity/Field schema: Scenario, Universe, FlowerOfLifeMultiverse
│   ├── models/
│   │   ├── mainstream/         # Verified standard physics solvers (Newtonian, NFW, GR, EM)
│   │   ├── thinkers/           # Standalone verbatim Thinker modules (Essonne, Haramein, Pais, Danaan, VogelWillis, ...)
│   │   └── unified/            # UnifiedSynthesis composing all Thinker modules
│   └── graphics/               # Desktop-only (UMGEBUNG_BUILD_GUI=ON) Window, RenderDevice (Vulkan/NVRHI/ImGui)
├── scenes/
│   ├── scenarios/              # Standalone .scenario.umgebung experiments & benchmarks
│   ├── universes/              # Individual .umgebung Bubble Universe files
│   └── multiverses/            # Flower of Life .multiverse.umgebung manifests
├── src/                        # C++20 (.cpp) and CUDA (.cu) implementations
│   └── Main.cpp
├── submodules/
│   ├── research/               # Thinkers corpus, computation manuals, corpus builder
│   └── vcpkg/                  # C++ dependency manager (manifest mode)
├── tests/                      # CTest numerical verification, conservation, A/B & Lean 4 tests
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── GEMINI.md
└── README.md
```

---

## 5. Build & Engineering Conventions

- **Languages & Tooling:** C++20, CUDA 20, CMake (&ge; 3.21), Ninja, and `vcpkg` manifest mode (`vcpkg.json`).
- **CUDA Architectures:** Target `70 75 80 89` (Volta on CUDA &le; 12.x; Turing, Ampere, Ada Lovelace on CUDA 13.x+) to cover every system in the Hardware Topology Matrix.
- **HPC Portability:** Keep GUI/rendering packages (`glfw3`, `imgui`, `nvrhi`, Vulkan) strictly scoped to `UMGEBUNG_BUILD_GUI=ON` / the `vcpkg` `gui` feature, and avoid hardcoding desktop-only compiler paths in HPC presets.
- **Verification Discipline:** Every mainstream numerical solver and standalone Thinker module should ship with automated CTest checks (analytical error bounds, conservation invariants, and A/B divergence tests) and matching Lean 4 theorems in `formal/` free of `sorry`.
- **Documentation & Commenting Style (Maximum Verbosity):**
  - Use **maximally verbose, Doxygen-compatible comments** (`/** ... */` with `@brief`, `@details`, `@param`, `@return`, `@pre`, `@post`, `@note`, `@warning`) on every file, namespace, class, struct, enum, field, function, and CUDA kernel.
  - Provide thorough step-by-step inline comments explaining **what** each non-trivial block does, **why** that design/algorithm was chosen, physical units and governing equations (citing the corresponding textbook formula, `Thinkers` source, or Lean 4 theorem), CUDA thread/block/memory-layout mechanics, and edge cases/invariants.
  - Apply the same verbose documentation standard across C++20/CUDA sources, CMake scripts, `.umgebung` example scenes (where supported), and Lean 4 formal files (`/-! ... -/` module docs and `/-- ... -/` docstrings).
