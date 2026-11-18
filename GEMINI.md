# GEMINI.md: Umgebung Development & Architecture Guide

## 1. Project Overview & Core Philosophy

**Umgebung** is an interactive 2D/3D multi-scale reality simulation engine and computational physics testbed written in **C++20** and **CUDA**. It is designed to operate across three tightly coupled pillars:

1. **Mainstream Physics Baselines (`models/mainstream`):**
   - Verified, textbook physics solvers (e.g., Newtonian and Post-Newtonian N-body gravity, symplectic integrators, Maxwell electrodynamics, classical/quantum reference systems).
   - Serves as an academically rigorous verification suite (analytical benchmarks, conservation-law checks, numerical convergence, and CUDA/HPC scaling) and as a controlled baseline for side-by-side comparison against alternative theories.
2. **Unified Research Models (`models/unified`):**
   - A singular, unified computational model synthesizing concepts and mathematics **verbatim** (not merely as loose inspiration) from the `Thinkers` in the `submodules/research` submodule, including but not limited to:
     - Elena Danaan
     - Alex Collier
     - Nassim Haramein
     - Chris Essonne
     - Christopher Cooper
     - Dan Willis
     - Marcel Vogel
     - Salvatore Pais
     - Masaru Emoto
   - Encompasses nested multiverse-to-subatomic hierarchy (`.umgebung` scene files per universe), holographic/geometric structure (Flower of Life, the Void), non-linear time and gravity fields, and consciousness/vibrational density/Phryll/Love dynamics.
3. **Formal Mathematical Verification (`formal/` — Lean 4):**
   - Uses **Lean 4** (with Lake / optional Mathlib) as an offline formal specification, proof, and reference-oracle layer alongside C++/CUDA.
   - Explicitly separates **proven consequences** from **postulated axioms**, verifies dimensional consistency and algebraic/discretization properties (e.g., symplecticity and discrete conservation), and optionally exports verified constants/reference tests for CTest without introducing any runtime or build dependency on HPC clusters.

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
  - **Desktop Interactive (`UMGEBUNG_BUILD_GUI=ON`):** Full 2D/3D interactive viewport and multiverse hierarchy inspector using GLFW, Vulkan + NVRHI, and Dear ImGui.
  - **Headless HPC Batch (`UMGEBUNG_BUILD_GUI=OFF`):** Zero windowing/graphics dependencies; driven via CLI (`cxxopts`) and `.umgebung` scene files for batch sweeps, checkpoints, and multi-GPU cluster runs.
- **Pluggable Per-Scene Physics Profiles:**
  - Every universe in the multiverse maps to its own `.umgebung` scene file declaring its initial state, hierarchy, and active **Physics Profile** (`Mainstream`, `Unified`, or a custom composition of force/field modules), enabling reproducible A/B testing across models.
- **Separation of Formal Proofs and Compute:**
  - Lean 4 never runs inside GPU kernels or HPC batch jobs. Proof verification (`lake build`) is optional in CMake (`UMGEBUNG_VERIFY_PROOFS`, `OFF` by default) so headless HPC builds remain self-contained.

---

## 4. Repository Layout

```text
Umgebung/
├── assets/                     # Icons, shaders, and runtime resources
├── formal/                     # Lean 4 formal specifications & proofs (optional)
│   └── Umgebung/
│       ├── Mainstream/         # Standard physics axioms, conservation laws, symmetries
│       ├── Unified/            # Formalized models & explicit axioms from research/Thinkers
│       └── Numerical/          # Discretization, integrator symplecticity & stability proofs
├── include/umgebung/           # Public C++20 / CUDA headers (mirrored by src/)
│   ├── core/                   # EngineContext, SystemProfiler, logging, CLI, timing
│   ├── compute/                # CUDA context, streams, memory, multi-GPU dispatch
│   ├── scene/                  # Multiverse/Universe hierarchy & .umgebung serialization
│   ├── models/
│   │   ├── mainstream/         # Verified standard physics solvers & benchmarks
│   │   └── unified/            # Unified research models & consciousness/field systems
│   └── graphics/               # Desktop-only (UMGEBUNG_BUILD_GUI=ON) Vulkan/NVRHI/ImGui
├── scenes/                     # Example .umgebung scene files (Mainstream & Unified)
├── src/                        # C++20 (.cpp) and CUDA (.cu) implementations
│   └── Main.cpp
├── submodules/
│   ├── research/               # Thinkers corpus, computation manuals, corpus builder
│   └── vcpkg/                  # C++ dependency manager (manifest mode)
├── tests/                      # CTest numerical verification, conservation & oracle tests
├── CMakeLists.txt
├── CMakePresets.json
├── vcpkg.json
├── GEMINI.md
└── README.md
```

---

## 5. Build & Engineering Conventions

- **Languages & Tooling:** C++20, CUDA 20, CMake (&ge; 3.21), Ninja, and `vcpkg` manifest mode (`vcpkg.json`).
- **CUDA Architectures:** Target `70 75 80 89` (Volta, Turing, Ampere, Ada Lovelace) to cover every system in the Hardware Topology Matrix.
- **HPC Portability:** Keep GUI/rendering packages (`glfw3`, `imgui`, `nvrhi`, Vulkan) strictly scoped to `UMGEBUNG_BUILD_GUI=ON` / the `vcpkg` `gui` feature, and avoid hardcoding desktop-only compiler paths in HPC presets.
- **Verification Discipline:** Every mainstream numerical solver should ship with automated CTest checks (analytical error bounds and conservation invariants) and, where applicable, matching Lean 4 theorems in `formal/` free of `sorry`.
- **Documentation & Commenting Style (Maximum Verbosity):**
  - Use **maximally verbose, Doxygen-compatible comments** (`/** ... */` with `@brief`, `@details`, `@param`, `@return`, `@pre`, `@post`, `@note`, `@warning`) on every file, namespace, class, struct, enum, field, function, and CUDA kernel.
  - Provide thorough step-by-step inline comments explaining **what** each non-trivial block does, **why** that design/algorithm was chosen, physical units and governing equations (citing the corresponding textbook formula, `Thinkers` source, or Lean 4 theorem), CUDA thread/block/memory-layout mechanics, and edge cases/invariants.
  - Apply the same verbose documentation standard across C++20/CUDA sources, CMake scripts, `.umgebung` example scenes (where supported), and Lean 4 formal files (`/-! ... -/` module docs and `/-- ... -/` docstrings).
