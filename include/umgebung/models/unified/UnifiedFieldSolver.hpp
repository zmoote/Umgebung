/**
 * @file UnifiedFieldSolver.hpp
 * @brief Pillar 2 (Unified Research Models): FP64 CUDA unified dynamical solver
 *        synthesizing verbatim mathematics from `submodules/research/Thinkers/`.
 *
 * @details
 * Implements `umgebung::models::IPhysicsSolver` in IEEE-754 double precision
 * (`double`) on `compute::CUDAContext` using Structure-of-Arrays (SoA)
 * `compute::DeviceBuffer<double>` allocations.
 *
 * Synthesized physical terms evaluated per step on the GPU:
 *   1. **Gravitational Dynamics:**
 *        - Standard pairwise Newtonian gravity between bodies, OR
 *        - **Chris Essonne's 2D Galactic Aether Vortex (`10100.pdf` Eq. 7 & 11)**
 *          around the primary galactic core (`body[0]`) when `aetherVortexEnabled`
 *          is true, reproducing flat periphery rotation curves without dark matter.
 *   2. **Love / Phryll Resonant Coherence Force (Danaan / Willis / Vogel):**
 *        - Strictly antisymmetric pairwise coupling ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$)
 *          scaled by `loveBindingCoupling * coherence_i * coherence_j * (phryll_i + phryll_j)`,
 *          preserving total linear momentum identically.
 *   3. **Per-Body Vacuum Pressure, Local Light Speed & Non-Linear Proper Time:**
 *        - Evaluates **Chris Essonne's** local vacuum pressure ratio
 *          $P_i / P_0 = 1 - \sum_{j \neq i} \frac{2 G m_j}{c_{\text{ref}}^2 r_{ij}}$
 *          (`8052.pdf` Eq. 21), local speed of light $c'_i / c = \sqrt{\max(P_i/P_0, 0)}$
 *          (`8052.pdf` Eq. 27), **Nassim Haramein's** holographic surface-to-volume
 *          ratio $\phi_i = 2 \ell_P / R_i$ (`Haramein342013PRRI3363.pdf`), and
 *          advances each entity's local proper time $\tau_i$ via
 *          `ComputeLocalTimeRate(P_i/P_0, D_i, D_0)` when `nonLinearTimeEnabled` is true.
 *   4. **Symplectic Integration:**
 *        - Uses 2nd-order symplectic Velocity Verlet (or Explicit Euler if configured),
 *          conserving total mechanical energy cleanly for Hamiltonian unified scenes.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace umgebung::models::unified {

    /**
     * @class UnifiedFieldSolver
     * @brief FP64 CUDA unified physics solver implementing `IPhysicsSolver` for
     *        `PhysicsProfileType::Unified` and `PhysicsProfileType::Custom` scenes.
     */
    class UnifiedFieldSolver final : public IPhysicsSolver {
    public:
        /** @brief Fixed shared-memory tile dimension (threads per block) for pairwise field tiling. */
        static constexpr int kFieldTileSize = 256;

        UnifiedFieldSolver() = default;
        ~UnifiedFieldSolver() override;

        UnifiedFieldSolver(const UnifiedFieldSolver&) = delete;
        UnifiedFieldSolver& operator=(const UnifiedFieldSolver&) = delete;
        UnifiedFieldSolver(UnifiedFieldSolver&&) = delete;
        UnifiedFieldSolver& operator=(UnifiedFieldSolver&&) = delete;

        /**
         * @brief Allocates FP64 SoA GPU buffers under `context`'s VRAM budget,
         *        uploads classical + unified entity properties from `scene`,
         *        computes initial accelerations & vacuum/holographic fields, and
         *        captures baseline total energy $E_0$.
         */
        [[nodiscard]] bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) override;

        /** @brief Synchronizes stream and releases all tracked GPU device buffers. */
        void Shutdown() noexcept override;

        /** @brief Enqueues one unified FP64 CUDA integration step of size `dt`. */
        [[nodiscard]] bool Step(double dt) override;

        /** @brief Enqueues `stepCount` steps and synchronizes the CUDA stream once. */
        [[nodiscard]] bool StepMultiple(std::uint64_t stepCount, double dt) override;

        /** @brief Synchronizes the solver's CUDA stream with the host. */
        [[nodiscard]] bool Synchronize() const noexcept override;

        /** @brief Downloads live positions and velocities into `outBodies`. */
        [[nodiscard]] bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) override;

        /**
         * @brief Downloads live GPU state & vacuum/proper-time buffers and populates
         *        `SolverDiagnostics` with both classical conservation invariants
         *        and Pillar 2 unified field telemetry (`hasUnifiedTelemetry = true`).
         */
        [[nodiscard]] bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) override;

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Unified Thinkers Field Solver (FP64 CUDA)";
        }

        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override {
            return m_profileType;
        }

        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override {
            return m_integrator;
        }

        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_bodyCount; }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

        /** @brief Downloads the per-body accumulated local proper times $\tau_i$ from the GPU. */
        [[nodiscard]] bool GetLocalProperTimes(std::vector<double>& outProperTimes);

        /** @brief Downloads the per-body Essonne vacuum pressure ratios $P_i / P_0$ from the GPU. */
        [[nodiscard]] bool GetVacuumPressureRatios(std::vector<double>& outPressureRatios);

    private:
        [[nodiscard]] bool LaunchUnifiedFieldKernel();
        [[nodiscard]] bool DownloadStateToStaging();

        compute::CUDAContext* m_context{nullptr};
        cudaStream_t m_stream{nullptr};

        bool m_initialized{false};
        std::size_t m_bodyCount{0U};
        scene::PhysicsProfileType m_profileType{scene::PhysicsProfileType::Unified};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};

        // Cached governing parameters from `UniverseScene::physicsProfile`.
        double m_gravitationalConstant{1.0};
        double m_softeningLength{0.0};
        scene::UnifiedFieldParams m_unifiedParams{};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        // GPU SoA FP64 Device Buffers (classical + unified per-entity state).
        compute::DeviceBuffer<double> m_posX;
        compute::DeviceBuffer<double> m_posY;
        compute::DeviceBuffer<double> m_posZ;

        compute::DeviceBuffer<double> m_velX;
        compute::DeviceBuffer<double> m_velY;
        compute::DeviceBuffer<double> m_velZ;

        compute::DeviceBuffer<double> m_accX;
        compute::DeviceBuffer<double> m_accY;
        compute::DeviceBuffer<double> m_accZ;

        compute::DeviceBuffer<double> m_mass;
        compute::DeviceBuffer<double> m_radius;
        compute::DeviceBuffer<double> m_vibrationalDensity;
        compute::DeviceBuffer<double> m_coherence;
        compute::DeviceBuffer<double> m_phryllCharge;
        compute::DeviceBuffer<double> m_vacuumPressureRatio;
        compute::DeviceBuffer<double> m_localProperTime;

        // Reusable Host Staging Buffers.
        std::vector<double> m_hostPosX;
        std::vector<double> m_hostPosY;
        std::vector<double> m_hostPosZ;
        std::vector<double> m_hostVelX;
        std::vector<double> m_hostVelY;
        std::vector<double> m_hostVelZ;
        std::vector<double> m_hostMass;
        std::vector<double> m_hostRadius;
        std::vector<double> m_hostDensity;
        std::vector<double> m_hostCoherence;
        std::vector<double> m_hostPhryll;
        std::vector<double> m_hostPressureRatio;
        std::vector<double> m_hostProperTime;
    };

} // namespace umgebung::models::unified
