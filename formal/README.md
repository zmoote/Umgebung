# Umgebung Formal Verification Layer (Lean 4)

This directory houses the **Lean 4** formal specifications, derivations, and numerical property proofs for Umgebung.

## Purpose

Lean 4 does **not** run inside the C++/CUDA simulation loop and is **not** required on headless HPC clusters. Instead, it serves three roles:

1. **Explicit Axioms & Consistency:** Formalize the governing equations, units/dimensions, and conservation laws for both `Mainstream` and `Unified` physics profiles so every assumption is explicit and machine-checked.
2. **Discretization & Integrator Proofs (`Numerical/`):** Prove mathematical properties of numerical schemes implemented in CUDA/C++ (e.g., symplecticity, time-reversibility, discrete momentum/energy invariants, quaternion/spinor identities).
3. **Reference Oracles & Codegen:** Optionally validate C++/CUDA numerical outputs against exact Lean reference computations in `tests/`, or generate verified constants/model definitions.

## Layout

```text
formal/
└── Umgebung/
    ├── Mainstream/   # Standard physics laws, symmetries, analytical benchmarks
    ├── Unified/      # Formalized definitions & axioms from submodules/research
    └── Numerical/    # Integrator & discretization proofs
```

## Planned Setup

Once `elan` / Lean 4 is installed locally:
- Initialize a Lake project (`lakefile.toml` + `lean-toolchain`) in `formal/`.
- Wire an optional CMake flag (`UMGEBUNG_VERIFY_PROOFS`, `OFF` by default) to run `lake build` during local builds and CI without affecting HPC builds.
