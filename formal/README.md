# Umgebung Formal Verification Layer (Lean 4)

This directory houses the **Lean 4** (`v4.36.0`) formal specifications, explicit postulated axioms, and **100% `sorry`-free** proofs for Umgebung.

## Architecture & Proven Theorem Catalog

All proofs are parameterized over a self-contained commutative field (`Umgebung.Algebra.ScalarField`, characteristic $\neq 2$) and 3D vector space (`Vec3`), compiling in seconds via pure Lean 4 core with zero external Mathlib cache dependencies.

```text
formal/
├── lean-toolchain                             # Pinned Lean 4.36.0 toolchain
├── lakefile.toml                              # Lake library manifest (`Umgebung`)
├── Umgebung.lean                              # Root library module
└── Umgebung/
    ├── Algebra/
    │   └── ScalarField.lean                   # Field axioms & 3D vector algebra (`Vec3`)
    ├── Numerical/
    │   ├── VelocityVerlet.lean                # Time-reversibility, momentum & symplecticity
    │   └── FdtdStability.lean                 # 2D Yee FDTD discrete div(B)=0, CFL ceiling & Poynting norm
    ├── Mainstream/
    │   ├── NewtonianGravity.lean              # Newton 3rd law, torque cancellation & Kepler virial
    │   └── DarkMatterNFW.lean                 # NFW halo scale-radius identity, zero torque & mass scaling
    └── Unified/
        ├── ThinkersModels.lean                # Haramein, Essonne, Pais, Danaan/Willis/Vogel
        ├── FlowerOfLife.lean                  # Flower of Life equilateral Vesica Piscis, Coriolis zero power, Pais & Vogel bounds
        └── TitorC204.lean                     # GE C204 Tipler/van Stockum, ADM metric horizons, Kerr-Newman & Borda consensus
```

### Key Machine-Checked Theorems (`sorry`-Free)
- **Numerical Integrator (`Umgebung.Numerical.VelocityVerlet`):**
  - `verletStep_time_reversible`: `verletStep accel (-dt) (verletStep accel dt s) = s`
  - `verletTwoBody_conserves_linear_momentum` & `verletNBody_conserves_linear_momentum`
  - `verletStep_symplectic_jacobian_det_one`: canonical phase-space Jacobian $\det(J_{\text{VV}}) = 1$
- **Pillar 1 Mainstream (`Umgebung.Mainstream.NewtonianGravity` & `DarkMatterNFW`):**
  - `newtonianPairForce_antisymm` & `newtonianPairForce_sum_zero`
  - `centralForce_torque_cancels`: $\mathbf{r}_i \times \mathbf{F}_{ij} + \mathbf{r}_j \times \mathbf{F}_{ji} = \mathbf{0}$
  - `kepler_circular_virial_energy`: $T + U = -\frac{G M m}{2 r}$
  - `nfwDensity_at_scale_radius`: $\rho_{\text{NFW}}(r_s) = \rho_0 / 4$
  - `nfwHaloForce_zero_torque`: $\mathbf{r} \times \mathbf{F}_{\text{NFW}}(\mathbf{r}) = \mathbf{0}$
  - `nfwEnclosedMass_linear_in_rho0`: linear scaling of $M_{\text{NFW}}(r)$ with $\rho_0$
- **Pillar 2 Unified (`Umgebung.Unified.ThinkersModels` & `FlowerOfLife`):**
  - **Haramein (2013):** `haramein_phi_closed_form`, `haramein_holographic_duality` ($m_h(r) \cdot m_p'(r) = 2 m_\ell^2$), `haramein_schwarzschild_equivalence`, `harameinCoriolis_zero_power` ($\mathbf{v} \cdot \mathbf{a}_{\text{Cor}} = 0$)
  - **Essonne (`8052.pdf` & `10100.pdf`):** `essonne_zero_central_pressure_at_3_2_Rs`, `essonne_true_horizon_2Rs`, `essonne_flat_galaxy_rotation_velocity`
  - **Pais (`US10144532.pdf`):** `pais_poynting_flux_superposition`, `paisInertialFactor_inactive`, `paisInertialFactor_mul_amplification`
  - **Danaan / Willis / Vogel:** `lovePhryllPairForce_antisymm`, `lovePhryllPairForce_torque_cancels`, `flowerOfLife_vesicaPiscis_equilateral`, `flowerOfLife_ring1_1_2_equilateral`, `vogelSaturationRatio_at_threshold`
  - **John Titor / GE C204 (`TitorC204.lean`):** `vanStockum_interior_coordinate_condition`, `vanStockum_ctl_boundary_at_inv_a`, `c204_adm_g00_sign_duality`, `c204_minkowski_exterior_limit`, `c204_null_time_horizon_exact`, `vanDenBroeck_core_shaping_unity`, `kerrNewman_extremal_double_root`, `kerrNewman_schwarzschild_limit`, `c204_potential_recovery_identity`, `c204_borda_unanimous_3_exact`, `c204_borda_unanimous_2_exact`

## Running Verification Locally

```bash
# Direct Lake verification
lake --dir formal build

# Integrated CMake + CTest verification (Desktop local builds)
cmake --preset linux-desktop-debug -DUMGEBUNG_VERIFY_PROOFS=ON
cmake --build --preset linux-desktop-debug --target umgebung_verify_proofs
ctest --preset linux-desktop-debug
```

Headless HPC builds (`hpc-headless-debug` / `hpc-headless-release`) keep `UMGEBUNG_VERIFY_PROOFS=OFF` by default and have zero dependency on `elan` or `lake`.
