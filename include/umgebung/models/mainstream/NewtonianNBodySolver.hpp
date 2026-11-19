/**
 * @file NewtonianNBodySolver.hpp
 * @brief Pillar 1 (Mainstream Physics): Double-precision (FP64) CUDA Newtonian
 *        N-body gravitational solver with symplectic Velocity Verlet integration.
 *
 * @details
 * Implements a textbook, academics-ready direct pairwise Newtonian gravitational
 * solver in IEEE-754 double precision (`double`) backed by Structure-of-Arrays
 * (SoA) GPU buffers (`compute::DeviceBuffer<double>`) and shared-memory tiled
 * CUDA kernels.
 *
 * Governing equations (for $N$ point masses $m_0, \dots, m_{N-1}$):
 *   - Gravitational acceleration on body $i$:
 *     $$\mathbf{a}_i = G \sum_{\substack{j = 0 \\ j \neq i}}^{N - 1} m_j
 *       \frac{\mathbf{r}_j - \mathbf{r}_i}{\left(\|\mathbf{r}_j - \mathbf{r}_i\|^2 + \varepsilon^2\right)^{3/2}}$$
 *   - 2nd-order symplectic Velocity Verlet (Kick-Drift-Kick) step of size $\Delta t$:
 *     1. $\mathbf{v}_i\!\left(t + \tfrac{\Delta t}{2}\right) = \mathbf{v}_i(t) + \tfrac{1}{2}\Delta t\,\mathbf{a}_i(t)$
 *     2. $\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \Delta t\,\mathbf{v}_i\!\left(t + \tfrac{\Delta t}{2}\right)$
 *     3. $\mathbf{a}_i(t + \Delta t) = \mathbf{a}(\mathbf{r}(t + \Delta t))$
 *     4. $\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i\!\left(t + \tfrac{\Delta t}{2}\right) + \tfrac{1}{2}\Delta t\,\mathbf{a}_i(t + \Delta t)$
 *
 * Verified conservation invariants computed via `ComputeDiagnostics()`:
 *   - Total mechanical energy $E = T + U$,
 *   - Total linear momentum $\mathbf{P} = \sum_i m_i \mathbf{v}_i$,
 *   - Total angular momentum $\mathbf{L} = \sum_i m_i (\mathbf{r}_i \times \mathbf{v}_i)$,
 *   - Center of mass $\mathbf{R}_{\text{cm}} = \frac{\sum_i m_i \mathbf{r}_i}{\sum_i m_i}$.
 *
 * @note
 * Zero GUI dependencies; compiles and runs identically in Desktop Interactive
 * GUI mode and Headless HPC batch mode.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @namespace umgebung::models::mainstream
 * @brief Pillar 1 verified standard-physics computational solvers and benchmarks.
 */
namespace umgebung::models::mainstream {

    /**
     * @struct NBodyDiagnostics
     * @brief Classical conservation invariants and energy telemetry snapshot
     *        computed from the active GPU state.
     */
    struct NBodyDiagnostics {
        /** @brief Total number of integration steps completed since initialization. */
        std::uint64_t stepCount{0ULL};

        /** @brief Accumulated simulation coordinate time $t$ in scene time units. */
        double simulationTime{0.0};

        /** @brief Total kinetic energy $T = \frac{1}{2} \sum_i m_i \|\mathbf{v}_i\|^2$. */
        double kineticEnergy{0.0};

        /** @brief Total gravitational potential energy $U = -\sum_{i < j} \frac{G m_i m_j}{\sqrt{r_{ij}^2 + \varepsilon^2}}$. */
        double potentialEnergy{0.0};

        /** @brief Total mechanical energy $E = T + U$. */
        double totalEnergy{0.0};

        /** @brief Baseline total mechanical energy $E_0$ captured at `t = 0`. */
        double initialTotalEnergy{0.0};

        /** @brief Dimensionless relative energy drift $|E - E_0| / \max(|E_0|, 10^{-30})$. */
        double relativeEnergyDrift{0.0};

        /** @brief Total linear momentum vector $\mathbf{P} = \sum_i m_i \mathbf{v}_i$. */
        scene::Vec3d totalLinearMomentum{};

        /** @brief Total angular momentum vector $\mathbf{L} = \sum_i m_i (\mathbf{r}_i \times \mathbf{v}_i)$. */
        scene::Vec3d totalAngularMomentum{};

        /** @brief System center-of-mass position $\mathbf{R}_{\text{cm}}$. */
        scene::Vec3d centerOfMass{};
    };

    /**
     * @class NewtonianNBodySolver
     * @brief GPU-accelerated FP64 Newtonian N-body gravitational integrator
     *        implementing `umgebung::models::IPhysicsSolver` with SoA
     *        `compute::DeviceBuffer<double>` arrays on a `CUDAContext`.
     */
    class NewtonianNBodySolver final : public IPhysicsSolver {
    public:
        /** @brief Fixed shared-memory tile dimension (threads per block) for pairwise gravity tiling. */
        static constexpr int kGravityTileSize = 256;

        /** @brief Constructs an uninitialized solver; call `Initialize()` before stepping. */
        NewtonianNBodySolver() = default;

        /** @brief Destructor; releases all owned GPU `DeviceBuffer<double>` allocations cleanly. */
        ~NewtonianNBodySolver() override;

        // Non-copyable; move-only semantics omitted because solver holds active context reference.
        NewtonianNBodySolver(const NewtonianNBodySolver&) = delete;
        NewtonianNBodySolver& operator=(const NewtonianNBodySolver&) = delete;
        NewtonianNBodySolver(NewtonianNBodySolver&&) = delete;
        NewtonianNBodySolver& operator=(NewtonianNBodySolver&&) = delete;

        /**
         * @brief Allocates FP64 SoA GPU buffers under `context`'s VRAM budget,
         *        uploads initial body state from `scene`, computes initial
         *        accelerations $\mathbf{a}(t_0)$ on the GPU, and records initial
         *        energy $E_0$ for drift tracking.
         *
         * @param context Initialized `compute::CUDAContext` owning the bound GPU & streams.
         * @param scene   Validated `.umgebung` `UniverseScene` supplying bodies and gravity config.
         * @return `true` on success; `false` on empty/invalid scene, VRAM budget
         *         overflow, or CUDA error.
         */
        [[nodiscard]] bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) override;

        /**
         * @brief Releases all owned GPU buffers and resets solver state.
         *        Must be called before `CUDAContext::Shutdown()` (guaranteed by RAII order).
         */
        void Shutdown() noexcept override;

        /**
         * @brief Advances the N-body system by one integration step of size `dt`
         *        on the GPU stream (asynchronously enqueued; call `Synchronize()`
         *        or `StepMultiple()` before reading results on host).
         *
         * @details
         * Supports both positive (`+dt`) and negative (`-dt`) finite non-zero
         * timesteps so time-reversibility invariants can be verified directly.
         *
         * @param dt Timestep $\Delta t$ in scene time units (must be finite and non-zero).
         * @return `true` if all CUDA kernels launched cleanly; `false` otherwise.
         */
        [[nodiscard]] bool Step(double dt) override;

        /**
         * @brief Enqueues `stepCount` consecutive integration steps of size `dt`
         *        on the GPU stream and synchronizes the stream once upon completion.
         *
         * @param stepCount Number of integration steps to execute.
         * @param dt        Timestep $\Delta t$ per step.
         * @return `true` if every launch and final stream synchronization succeeded.
         */
        [[nodiscard]] bool StepMultiple(std::uint64_t stepCount, double dt) override;

        /**
         * @brief Synchronizes the solver's CUDA stream with the host thread.
         *
         * @return `true` on `cudaSuccess`; `false` on device error.
         */
        [[nodiscard]] bool Synchronize() const noexcept override;

        /**
         * @brief Synchronizes the GPU stream and downloads the current positions
         *        and velocities into `outBodies`, preserving body metadata (`id`,
         *        `name`, `mass`, etc.) when `outBodies.size() == GetBodyCount()`.
         *
         * @param[in,out] outBodies Vector updated in place (or resized) with live GPU state.
         * @return `true` on success; `false` on CUDA download failure.
         */
        [[nodiscard]] bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) override;

        /**
         * @brief Synchronizes the GPU stream, downloads current SoA state, and
         *        computes exact FP64 energy, momentum, angular momentum, and
         *        center-of-mass diagnostics on the host.
         *
         * @param[out] outDiagnostics Populated `NBodyDiagnostics` snapshot.
         * @return `true` on success; `false` if uninitialized or CUDA transfer failed.
         */
        [[nodiscard]] bool ComputeDiagnostics(NBodyDiagnostics& outDiagnostics);

        /**
         * @brief `IPhysicsSolver` diagnostic adapter populating `SolverDiagnostics`.
         */
        [[nodiscard]] bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) override;

        /** @brief Returns `true` if GPU buffers are allocated and initial accelerations are ready. */
        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        /** @brief Returns the human-readable Pillar 1 solver label. */
        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Mainstream Newtonian N-Body (FP64 CUDA)";
        }

        /** @brief Returns `PhysicsProfileType::Mainstream`. */
        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override {
            return scene::PhysicsProfileType::Mainstream;
        }

        /** @brief Returns the number of simulated gravitating bodies $N$. */
        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_bodyCount; }

        /** @brief Returns the total number of completed integration steps. */
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }

        /** @brief Returns accumulated simulation time $t$ in scene time units. */
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

        /** @brief Returns the active gravitational constant $G$. */
        [[nodiscard]] double GetGravitationalConstant() const noexcept { return m_gravitationalConstant; }

        /** @brief Returns the active Plummer softening length $\varepsilon$. */
        [[nodiscard]] double GetSofteningLength() const noexcept { return m_softeningLength; }

        /** @brief Returns the active numerical integrator (`VelocityVerlet` or `EulerExplicit`). */
        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override { return m_integrator; }

    private:
        /**
         * @brief Launches the shared-memory tiled FP64 pairwise gravity kernel
         *        on `m_stream` to refresh `(d_accX, d_accY, d_accZ)` from current
         *        positions `(d_posX, d_posY, d_posZ)`.
         */
        [[nodiscard]] bool LaunchAccelerationKernel();

        /**
         * @brief Downloads all SoA arrays (`pos`, `vel`, `mass`) into the
         *        reusable host staging vectors `m_host*` after synchronizing `m_stream`.
         */
        [[nodiscard]] bool DownloadStateToStaging();

        /** @brief Non-owning pointer to the active `CUDAContext`. */
        compute::CUDAContext* m_context{nullptr};

        /** @brief Dedicated non-blocking CUDA stream borrowed from `m_context`. */
        cudaStream_t m_stream{nullptr};

        /** @brief True after `Initialize()` succeeds. */
        bool m_initialized{false};

        /** @brief Number of bodies $N$ currently loaded on the GPU. */
        std::size_t m_bodyCount{0U};

        /** @brief Configured gravitational constant $G$. */
        double m_gravitationalConstant{1.0};

        /** @brief Configured Plummer softening length $\varepsilon$. */
        double m_softeningLength{0.0};

        /** @brief Configured numerical time-integration scheme. */
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};

        /** @brief Completed integration step counter. */
        std::uint64_t m_stepCount{0ULL};

        /** @brief Accumulated simulation coordinate time $t$. */
        double m_simulationTime{0.0};

        /** @brief Baseline total mechanical energy $E_0$ recorded at `Initialize()`. */
        double m_initialTotalEnergy{0.0};

        // ---------------------------------------------------------------------
        // GPU Structure-of-Arrays (SoA) FP64 Device Buffers
        // ---------------------------------------------------------------------
        // Contiguous per-component arrays guarantee coalesced 128-byte warp
        // memory transactions on Volta, Turing, Ampere, and Ada Lovelace GPUs.
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

        // ---------------------------------------------------------------------
        // Reusable Host Staging Arrays (Avoid Per-Frame Heap Allocations)
        // ---------------------------------------------------------------------
        std::vector<double> m_hostPosX;
        std::vector<double> m_hostPosY;
        std::vector<double> m_hostPosZ;
        std::vector<double> m_hostVelX;
        std::vector<double> m_hostVelY;
        std::vector<double> m_hostVelZ;
        std::vector<double> m_hostMass;
    };

} // namespace umgebung::models::mainstream
