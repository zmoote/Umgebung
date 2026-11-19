/**
 * @file NewtonianNBodySolver.cu
 * @brief CUDA FP64 implementation of Pillar 1's textbook Newtonian N-body
 *        gravitational solver with shared-memory tiling and Velocity Verlet.
 *
 * @details
 * Implements three device kernels plus host orchestration for
 * `umgebung::models::mainstream::NewtonianNBodySolver`:
 *   1. `ComputeAccelerationsTiledKernel`:
 *      Computes all-pairs Newtonian gravitational acceleration in IEEE-754
 *      double precision (`double`) using cooperative `__shared__` memory
 *      tiling (`kTileSize = 256`, 8 warps/block, 8 KiB shared memory/block).
 *      Uses strict `j != i` exclusion so unsoftened ($\varepsilon = 0$)
 *      analytical problems never produce self-force `NaN`s, and exact
 *      `1.0 / sqrt(distSq)` for full IEEE-754 ULP precision.
 *   2. `VelocityVerletKickDriftKernel`:
 *      Executes the first half-kick + full position drift of Kick-Drift-Kick
 *      Velocity Verlet:
 *        $$\mathbf{v}_{i}^{n + 1/2} = \mathbf{v}_i^n + \frac{\Delta t}{2}\mathbf{a}_i^n, \qquad
 *          \mathbf{r}_i^{n+1} = \mathbf{r}_i^n + \Delta t\,\mathbf{v}_i^{n + 1/2}$$
 *   3. `VelocityVerletFinalKickKernel`:
 *      Completes the symmetric second half-kick using newly evaluated
 *      accelerations $\mathbf{a}_i^{n+1}$:
 *        $$\mathbf{v}_i^{n+1} = \mathbf{v}_i^{n + 1/2} + \frac{\Delta t}{2}\mathbf{a}_i^{n+1}$$
 *   4. `EulerExplicitStepKernel`:
 *      Optional 1st-order explicit Euler step for pedagogical side-by-side
 *      comparison showing why symplectic integrators are required for orbits.
 */

#include "umgebung/models/mainstream/NewtonianNBodySolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"

#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::mainstream {

    namespace {

        /**
         * @brief Shared-memory tiled FP64 pairwise Newtonian gravity kernel.
         *
         * @details
         * Thread mapping:
         *   - Global thread index `i = blockIdx.x * blockDim.x + threadIdx.x`
         *     corresponds to target body $i \in [0, N)$.
         *   - All threads in the block (including tail threads where `i >= N`)
         *     participate in `__syncthreads()` barriers around each shared-memory
         *     tile load to guarantee deadlock-free execution for arbitrary $N$.
         *   - Global memory reads are reduced by a factor of `kTileSize` (256x)
         *     relative to a naive global-memory $O(N^2)$ loop.
         *
         * @param posX, posY, posZ Device pointers to body Cartesian coordinates.
         * @param mass             Device pointer to body gravitational masses $m_j$.
         * @param accX, accY, accZ Output device pointers populated with $\mathbf{a}_i$.
         * @param bodyCount        Total number of bodies $N$.
         * @param G                Gravitational constant $G$.
         * @param softeningSq      Squared Plummer softening length $\varepsilon^2 \ge 0$.
         */
        template <int kTileSize>
        __global__ void ComputeAccelerationsTiledKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
            double* __restrict__ accX,
            double* __restrict__ accY,
            double* __restrict__ accZ,
            std::size_t bodyCount,
            double G,
            double softeningSq
        ) {
            // Cooperative block-local tile cache in fast on-chip SRAM (8 KiB for kTileSize = 256).
            __shared__ double sPosX[kTileSize];
            __shared__ double sPosY[kTileSize];
            __shared__ double sPosZ[kTileSize];
            __shared__ double sMass[kTileSize];

            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);

            // Load target body `i` into thread-private registers once before the tile sweep.
            double xi = 0.0;
            double yi = 0.0;
            double zi = 0.0;
            if (i < bodyCount) {
                xi = posX[i];
                yi = posY[i];
                zi = posZ[i];
            }

            double ax = 0.0;
            double ay = 0.0;
            double az = 0.0;

            // Sweep across all source bodies in chunks of `kTileSize`.
            for (std::size_t tileStart = 0U; tileStart < bodyCount; tileStart += static_cast<std::size_t>(kTileSize)) {
                const std::size_t loadIdx = tileStart + static_cast<std::size_t>(threadIdx.x);
                if (loadIdx < bodyCount) {
                    sPosX[threadIdx.x] = posX[loadIdx];
                    sPosY[threadIdx.x] = posY[loadIdx];
                    sPosZ[threadIdx.x] = posZ[loadIdx];
                    sMass[threadIdx.x] = mass[loadIdx];
                } else {
                    sPosX[threadIdx.x] = 0.0;
                    sPosY[threadIdx.x] = 0.0;
                    sPosZ[threadIdx.x] = 0.0;
                    sMass[threadIdx.x] = 0.0;
                }

                // Barrier 1: Ensure entire tile is visible to all threads in the block.
                __syncthreads();

                if (i < bodyCount) {
                    const std::size_t remaining = bodyCount - tileStart;
                    const std::size_t tileLimit =
                        (remaining < static_cast<std::size_t>(kTileSize))
                            ? remaining
                            : static_cast<std::size_t>(kTileSize);

                    for (std::size_t k = 0U; k < tileLimit; ++k) {
                        const std::size_t j = tileStart + k;
                        // Strictly skip self-interaction so exact zero-softening
                        // problems ($\varepsilon = 0$) never divide by zero.
                        if (j == i) {
                            continue;
                        }

                        const double dx = sPosX[k] - xi;
                        const double dy = sPosY[k] - yi;
                        const double dz = sPosZ[k] - zi;
                        const double distSq = (dx * dx) + (dy * dy) + (dz * dz) + softeningSq;

                        if (distSq > 0.0) {
                            // Exact IEEE-754 FP64 reciprocal distance cubed: 1 / (r^2 + eps^2)^(3/2).
                            const double invDist = 1.0 / sqrt(distSq);
                            const double invDist3 = invDist * invDist * invDist;
                            const double factor = sMass[k] * invDist3;

                            ax += dx * factor;
                            ay += dy * factor;
                            az += dz * factor;
                        }
                    }
                }

                // Barrier 2: Ensure all threads finished reading this tile before next overwrite.
                __syncthreads();
            }

            if (i < bodyCount) {
                accX[i] = G * ax;
                accY[i] = G * ay;
                accZ[i] = G * az;
            }
        }

        /**
         * @brief Velocity Verlet stage 1: half-step velocity kick followed by
         *        full-step position drift.
         *
         * @details
         * For every body $i < N$:
         *   $$\mathbf{v}_i \leftarrow \mathbf{v}_i + \tfrac{1}{2}\Delta t\,\mathbf{a}_i, \qquad
         *     \mathbf{r}_i \leftarrow \mathbf{r}_i + \Delta t\,\mathbf{v}_i$$
         */
        __global__ void VelocityVerletKickDriftKernel(
            double* __restrict__ posX,
            double* __restrict__ posY,
            double* __restrict__ posZ,
            double* __restrict__ velX,
            double* __restrict__ velY,
            double* __restrict__ velZ,
            const double* __restrict__ accX,
            const double* __restrict__ accY,
            const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }

            const double halfDt = 0.5 * dt;
            const double vxHalf = velX[i] + (halfDt * accX[i]);
            const double vyHalf = velY[i] + (halfDt * accY[i]);
            const double vzHalf = velZ[i] + (halfDt * accZ[i]);

            velX[i] = vxHalf;
            velY[i] = vyHalf;
            velZ[i] = vzHalf;

            posX[i] += dt * vxHalf;
            posY[i] += dt * vyHalf;
            posZ[i] += dt * vzHalf;
        }

        /**
         * @brief Velocity Verlet stage 3: symmetric second half-step velocity kick
         *        using the updated acceleration $\mathbf{a}(t + \Delta t)$.
         */
        __global__ void VelocityVerletFinalKickKernel(
            double* __restrict__ velX,
            double* __restrict__ velY,
            double* __restrict__ velZ,
            const double* __restrict__ accX,
            const double* __restrict__ accY,
            const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }

            const double halfDt = 0.5 * dt;
            velX[i] += halfDt * accX[i];
            velY[i] += halfDt * accY[i];
            velZ[i] += halfDt * accZ[i];
        }

        /**
         * @brief First-order explicit Euler step (non-symplectic baseline for
         *        pedagogical energy-drift comparisons against Velocity Verlet).
         */
        __global__ void EulerExplicitStepKernel(
            double* __restrict__ posX,
            double* __restrict__ posY,
            double* __restrict__ posZ,
            double* __restrict__ velX,
            double* __restrict__ velY,
            double* __restrict__ velZ,
            const double* __restrict__ accX,
            const double* __restrict__ accY,
            const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }

            posX[i] += dt * velX[i];
            posY[i] += dt * velY[i];
            posZ[i] += dt * velZ[i];

            velX[i] += dt * accX[i];
            velY[i] += dt * accY[i];
            velZ[i] += dt * accZ[i];
        }

    } // anonymous namespace

    NewtonianNBodySolver::~NewtonianNBodySolver() {
        Shutdown();
    }

    bool NewtonianNBodySolver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        Shutdown();

        if (!context.IsInitialized()) {
            spdlog::error("NewtonianNBodySolver::Initialize failed: CUDAContext is not initialized.");
            return false;
        }

        std::string validationErr;
        if (!scene.Validate(validationErr)) {
            spdlog::error("NewtonianNBodySolver::Initialize failed: invalid UniverseScene ({}).", validationErr);
            return false;
        }

        if (scene.bodies.empty()) {
            spdlog::error("NewtonianNBodySolver::Initialize failed: scene '{}' contains 0 bodies.", scene.universeId);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(0U);
        m_bodyCount = scene.bodies.size();
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_integrator = scene.physicsProfile.integrator;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        // ---------------------------------------------------------------------
        // Step 1: Allocate Budget-Tracked FP64 SoA Device Buffers
        // ---------------------------------------------------------------------
        if (!m_posX.Allocate(context, m_bodyCount) ||
            !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) ||
            !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) ||
            !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) ||
            !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) ||
            !m_mass.Allocate(context, m_bodyCount)) {
            spdlog::error("NewtonianNBodySolver: Failed allocating FP64 SoA GPU buffers for {} bodies.", m_bodyCount);
            Shutdown();
            return false;
        }

        // ---------------------------------------------------------------------
        // Step 2: Pack AoS `BodyEntity` Scene Data into Contiguous SoA Host Arrays
        // ---------------------------------------------------------------------
        m_hostPosX.resize(m_bodyCount);
        m_hostPosY.resize(m_bodyCount);
        m_hostPosZ.resize(m_bodyCount);
        m_hostVelX.resize(m_bodyCount);
        m_hostVelY.resize(m_bodyCount);
        m_hostVelZ.resize(m_bodyCount);
        m_hostMass.resize(m_bodyCount);

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const scene::BodyEntity& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x;
            m_hostPosY[i] = b.position.y;
            m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x;
            m_hostVelY[i] = b.velocity.y;
            m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;
        }

        // Upload initial state asynchronously onto `m_stream`.
        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) ||
            !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) ||
            !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) ||
            !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) ||
            !m_accX.ZeroFill(m_stream) ||
            !m_accY.ZeroFill(m_stream) ||
            !m_accZ.ZeroFill(m_stream)) {
            spdlog::error("NewtonianNBodySolver: Failed uploading initial scene state to GPU.");
            Shutdown();
            return false;
        }

        // ---------------------------------------------------------------------
        // Step 3: Compute Initial Accelerations a(t_0) & Baseline Energy E_0
        // ---------------------------------------------------------------------
        if (!LaunchAccelerationKernel() || !Synchronize()) {
            spdlog::error("NewtonianNBodySolver: Initial acceleration kernel failed.");
            Shutdown();
            return false;
        }

        m_initialized = true;

        NBodyDiagnostics initialDiag{};
        if (!ComputeDiagnostics(initialDiag)) {
            Shutdown();
            return false;
        }
        m_initialTotalEnergy = initialDiag.totalEnergy;

        spdlog::info(
            "NewtonianNBodySolver initialized for '{}' ({} bodies, integrator={}, G={:.6g}, eps={:.6g}, E0={:.10g}).",
            scene.name,
            m_bodyCount,
            scene::IntegratorTypeToString(m_integrator),
            m_gravitationalConstant,
            m_softeningLength,
            m_initialTotalEnergy
        );
        return true;
    }

    void NewtonianNBodySolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }

        m_posX.Release();
        m_posY.Release();
        m_posZ.Release();
        m_velX.Release();
        m_velY.Release();
        m_velZ.Release();
        m_accX.Release();
        m_accY.Release();
        m_accZ.Release();
        m_mass.Release();

        m_hostPosX.clear();
        m_hostPosY.clear();
        m_hostPosZ.clear();
        m_hostVelX.clear();
        m_hostVelY.clear();
        m_hostVelZ.clear();
        m_hostMass.clear();

        m_context = nullptr;
        m_stream = nullptr;
        m_initialized = false;
        m_bodyCount = 0U;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_initialTotalEnergy = 0.0;
    }

    bool NewtonianNBodySolver::LaunchAccelerationKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }

        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kGravityTileSize);
        if (gridSize == 0U) {
            return true;
        }

        const double softeningSq = m_softeningLength * m_softeningLength;
        ComputeAccelerationsTiledKernel<kGravityTileSize><<<gridSize, kGravityTileSize, 0, m_stream>>>(
            m_posX.Data(),
            m_posY.Data(),
            m_posZ.Data(),
            m_mass.Data(),
            m_accX.Data(),
            m_accY.Data(),
            m_accZ.Data(),
            m_bodyCount,
            m_gravitationalConstant,
            softeningSq
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool NewtonianNBodySolver::Step(double dt) {
        if (!m_initialized || m_context == nullptr) {
            spdlog::error("NewtonianNBodySolver::Step called on uninitialized solver.");
            return false;
        }

        if (!std::isfinite(dt) || dt == 0.0) {
            spdlog::error("NewtonianNBodySolver::Step requires a finite non-zero timestep (got {}).", dt);
            return false;
        }

        if (!m_context->MakeCurrent()) {
            return false;
        }

        const int blockSize = m_context->GetRecommendedBlockSize();
        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, blockSize);
        if (gridSize == 0U) {
            return true;
        }

        if (m_integrator == scene::IntegratorType::VelocityVerlet) {
            // Stage 1: Half-step velocity kick + full-step position drift.
            VelocityVerletKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }

            // Stage 2: Recompute pairwise gravitational accelerations a(t + dt).
            if (!LaunchAccelerationKernel()) {
                return false;
            }

            // Stage 3: Symmetric second half-step velocity kick.
            VelocityVerletFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            // Explicit Euler (non-symplectic reference).
            EulerExplicitStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }

            if (!LaunchAccelerationKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool NewtonianNBodySolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool NewtonianNBodySolver::Synchronize() const noexcept {
        if (m_context == nullptr || !m_context->IsInitialized()) {
            return false;
        }
        return m_context->SynchronizeStream(0U);
    }

    bool NewtonianNBodySolver::DownloadStateToStaging() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }

        if (!m_posX.CopyToHost(m_hostPosX, m_stream) ||
            !m_posY.CopyToHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyToHost(m_hostPosZ, m_stream) ||
            !m_velX.CopyToHost(m_hostVelX, m_stream) ||
            !m_velY.CopyToHost(m_hostVelY, m_stream) ||
            !m_velZ.CopyToHost(m_hostVelZ, m_stream)) {
            return false;
        }

        return Synchronize();
    }

    bool NewtonianNBodySolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        if (outBodies.size() != m_bodyCount) {
            outBodies.resize(m_bodyCount);
        }

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            outBodies[i].position = {m_hostPosX[i], m_hostPosY[i], m_hostPosZ[i]};
            outBodies[i].velocity = {m_hostVelX[i], m_hostVelY[i], m_hostVelZ[i]};
            outBodies[i].mass = m_hostMass[i];
        }
        return true;
    }

    bool NewtonianNBodySolver::ComputeDiagnostics(NBodyDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0;
        double potential = 0.0;
        double totalMass = 0.0;
        scene::Vec3d linearMomentum{};
        scene::Vec3d angularMomentum{};
        scene::Vec3d weightedPos{};

        const double softeningSq = m_softeningLength * m_softeningLength;

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const double mi = m_hostMass[i];
            const scene::Vec3d ri{m_hostPosX[i], m_hostPosY[i], m_hostPosZ[i]};
            const scene::Vec3d vi{m_hostVelX[i], m_hostVelY[i], m_hostVelZ[i]};

            totalMass += mi;
            kinetic += 0.5 * mi * vi.NormSquared();

            const scene::Vec3d pi = vi * mi;
            linearMomentum = linearMomentum + pi;
            angularMomentum = angularMomentum + scene::Vec3d::Cross(ri, pi);
            weightedPos = weightedPos + (ri * mi);

            // Sum unique unordered pairs (i < j) to compute exact gravitational potential energy U.
            for (std::size_t j = i + 1U; j < m_bodyCount; ++j) {
                const double dx = m_hostPosX[j] - ri.x;
                const double dy = m_hostPosY[j] - ri.y;
                const double dz = m_hostPosZ[j] - ri.z;
                const double dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz) + softeningSq);
                if (dist > 0.0) {
                    potential -= (m_gravitationalConstant * mi * m_hostMass[j]) / dist;
                }
            }
        }

        const double totalE = kinetic + potential;
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0)
            ? totalE
            : m_initialTotalEnergy;
        const double denom = std::max(std::abs(baselineE), 1.0e-30);

        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.kineticEnergy = kinetic;
        outDiagnostics.potentialEnergy = potential;
        outDiagnostics.totalEnergy = totalE;
        outDiagnostics.initialTotalEnergy = baselineE;
        outDiagnostics.relativeEnergyDrift = std::abs(totalE - baselineE) / denom;
        outDiagnostics.totalLinearMomentum = linearMomentum;
        outDiagnostics.totalAngularMomentum = angularMomentum;
        outDiagnostics.centerOfMass = (totalMass > 0.0)
            ? (weightedPos * (1.0 / totalMass))
            : scene::Vec3d{};

        return true;
    }

    bool NewtonianNBodySolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        NBodyDiagnostics d{};
        if (!ComputeDiagnostics(d)) {
            return false;
        }

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.stepCount = d.stepCount;
        outDiagnostics.simulationTime = d.simulationTime;
        outDiagnostics.kineticEnergy = d.kineticEnergy;
        outDiagnostics.potentialEnergy = d.potentialEnergy;
        outDiagnostics.totalEnergy = d.totalEnergy;
        outDiagnostics.initialTotalEnergy = d.initialTotalEnergy;
        outDiagnostics.relativeEnergyDrift = d.relativeEnergyDrift;
        outDiagnostics.totalLinearMomentum = d.totalLinearMomentum;
        outDiagnostics.totalAngularMomentum = d.totalAngularMomentum;
        outDiagnostics.centerOfMass = d.centerOfMass;
        outDiagnostics.hasUnifiedTelemetry = false;
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::mainstream
