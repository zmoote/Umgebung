/**
 * @file UnifiedFieldSolver.cu
 * @brief CUDA FP64 implementation of Pillar 2's unified field solver
 *        (`umgebung::models::unified::UnifiedFieldSolver`).
 *
 * @details
 * Implements GPU kernels evaluating the unified synthesis of:
 *   - Chris Essonne's 2D galactic aether-vortex radial field (`10100.pdf` Eq. 7 & 11)
 *     and vacuum pressure gradient / local light speed (`8052.pdf` Eq. 21, 27),
 *   - Nassim Haramein's holographic PSU surface-to-volume ratio $\phi$ (`Haramein342013PRRI3363.pdf`),
 *   - Elena Danaan / Dan Willis / Marcel Vogel's antisymmetric Love/Phryll resonant
 *     coherence force and gravity/vibrational-density coupled non-linear proper time.
 */

#include "umgebung/models/unified/UnifiedFieldSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/unified/UnifiedPhysicsEquations.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::unified {

    namespace {

        /**
         * @brief Shared-memory tiled FP64 unified acceleration & vacuum-pressure kernel.
         *
         * @details
         * For each target body $i \in [0, N)$:
         *   1. Computes pairwise Newtonian gravity (when `aetherVortexEnabled == false`)
         *      or pairwise interactions plus Chris Essonne's 2D galactic aether-vortex
         *      radial field centered on `body[0]` (`10100.pdf` Eq. 7 & 11) when
         *      `aetherVortexEnabled == true`.
         *   2. Adds strictly antisymmetric Love/Phryll resonant coherence force:
         *      $$\mathbf{F}_{ij}^{\text{LP}} = \kappa_{\text{love}} \, C_i C_j (Q_i + Q_j)
         *        \frac{\mathbf{r}_j - \mathbf{r}_i}{\left(\|\mathbf{r}_j - \mathbf{r}_i\|^2 + \varepsilon^2\right)^{3/2}}$$
         *      divided by $m_i$ (so $\mathbf{F}_{ij}^{\text{LP}} + \mathbf{F}_{ji}^{\text{LP}} = \mathbf{0}$,
         *      preserving total linear momentum identically).
         *   3. Evaluates Essonne's local vacuum pressure ratio:
         *      $$\frac{P_i}{P_0} = 1 - \sum_{j \neq i} \frac{2 G m_j}{c_{\text{ref}}^2 \sqrt{r_{ij}^2 + \varepsilon^2}}$$
         *      (`8052.pdf` Eq. 21).
         */
        template <int kTileSize>
        __global__ void ComputeUnifiedFieldKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
            const double* __restrict__ coherence,
            const double* __restrict__ phryll,
            double* __restrict__ accX,
            double* __restrict__ accY,
            double* __restrict__ accZ,
            double* __restrict__ vacuumPressureRatio,
            std::size_t bodyCount,
            double G,
            double softeningSq,
            double refLightSpeed,
            bool aetherVortexEnabled,
            double vortexRc,
            double vortexScaleH,
            double vortexGz,
            double loveCoupling
        ) {
            __shared__ double sPosX[kTileSize];
            __shared__ double sPosY[kTileSize];
            __shared__ double sPosZ[kTileSize];
            __shared__ double sMass[kTileSize];
            __shared__ double sCoh[kTileSize];
            __shared__ double sPhryll[kTileSize];

            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);

            double xi = 0.0;
            double yi = 0.0;
            double zi = 0.0;
            double mi = 1.0;
            double ci = 0.0;
            double qi = 0.0;
            if (i < bodyCount) {
                xi = posX[i];
                yi = posY[i];
                zi = posZ[i];
                mi = mass[i];
                ci = coherence[i];
                qi = phryll[i];
            }

            double ax = 0.0;
            double ay = 0.0;
            double az = 0.0;
            double pressureDepletion = 0.0;
            const double invC2 = (refLightSpeed > 0.0) ? (1.0 / (refLightSpeed * refLightSpeed)) : 0.0;

            for (std::size_t tileStart = 0U; tileStart < bodyCount; tileStart += static_cast<std::size_t>(kTileSize)) {
                const std::size_t loadIdx = tileStart + static_cast<std::size_t>(threadIdx.x);
                if (loadIdx < bodyCount) {
                    sPosX[threadIdx.x] = posX[loadIdx];
                    sPosY[threadIdx.x] = posY[loadIdx];
                    sPosZ[threadIdx.x] = posZ[loadIdx];
                    sMass[threadIdx.x] = mass[loadIdx];
                    sCoh[threadIdx.x] = coherence[loadIdx];
                    sPhryll[threadIdx.x] = phryll[loadIdx];
                } else {
                    sPosX[threadIdx.x] = 0.0;
                    sPosY[threadIdx.x] = 0.0;
                    sPosZ[threadIdx.x] = 0.0;
                    sMass[threadIdx.x] = 0.0;
                    sCoh[threadIdx.x] = 0.0;
                    sPhryll[threadIdx.x] = 0.0;
                }

                __syncthreads();

                if (i < bodyCount) {
                    const std::size_t remaining = bodyCount - tileStart;
                    const std::size_t tileLimit =
                        (remaining < static_cast<std::size_t>(kTileSize))
                            ? remaining
                            : static_cast<std::size_t>(kTileSize);

                    for (std::size_t k = 0U; k < tileLimit; ++k) {
                        const std::size_t j = tileStart + k;
                        if (j == i) {
                            continue;
                        }

                        const double dx = sPosX[k] - xi;
                        const double dy = sPosY[k] - yi;
                        const double dz = sPosZ[k] - zi;
                        const double distSq = (dx * dx) + (dy * dy) + (dz * dz) + softeningSq;

                        if (distSq > 0.0) {
                            const double dist = sqrt(distSq);
                            const double invDist = 1.0 / dist;
                            const double invDist3 = invDist * invDist * invDist;

                            // Accumulate Essonne vacuum pressure depletion `2 G m_j / (c^2 r_ij)` (`8052.pdf` Eq. 21).
                            pressureDepletion += (2.0 * G * sMass[k] * invC2) * invDist;

                            // When Essonne's 2D galactic aether vortex is active, body 0 represents
                            // the flat cylindrical galactic core exerting Essonne's `10100.pdf` Eq. (7)/(11)
                            // radial vortex field on orbiting stars (i > 0), rather than isotropic 1/r^2.
                            if (aetherVortexEnabled && j == 0U && i > 0U) {
                                const double cylR = sqrt((dx * dx) + (dy * dy));
                                if (cylR > 0.0) {
                                    const double gr = ComputeEssonneAetherVortexRadialAccel(
                                        cylR,
                                        sMass[k],
                                        G,
                                        vortexRc,
                                        vortexScaleH,
                                        vortexGz
                                    );
                                    const double invCylR = 1.0 / cylR;
                                    ax += gr * (dx * invCylR);
                                    ay += gr * (dy * invCylR);
                                }
                            } else if (!(aetherVortexEnabled && i == 0U)) {
                                // Standard pairwise Newtonian gravity.
                                const double gravScale = G * sMass[k] * invDist3;
                                ax += dx * gravScale;
                                ay += dy * gravScale;
                                az += dz * gravScale;
                            }

                            // Antisymmetric Love/Phryll resonant coherence attraction:
                            // F_ij = loveCoupling * (C_i * C_j) * (Q_i + Q_j) * r_ij / |r_ij|^3.
                            if (loveCoupling != 0.0 && mi > 0.0) {
                                const double lpForceMag =
                                    loveCoupling * (ci * sCoh[k]) * (qi + sPhryll[k]) * invDist3;
                                const double lpAccelScale = lpForceMag / mi;
                                ax += dx * lpAccelScale;
                                ay += dy * lpAccelScale;
                                az += dz * lpAccelScale;
                            }
                        }
                    }
                }

                __syncthreads();
            }

            if (i < bodyCount) {
                accX[i] = ax;
                accY[i] = ay;
                accZ[i] = az;
                const double ratio = 1.0 - pressureDepletion;
                vacuumPressureRatio[i] = (ratio > 0.0) ? ratio : 0.0;
            }
        }

        /**
         * @brief Velocity Verlet stage 1 (half-kick + drift) + local proper-time advance.
         */
        __global__ void UnifiedKickDriftKernel(
            double* __restrict__ posX,
            double* __restrict__ posY,
            double* __restrict__ posZ,
            double* __restrict__ velX,
            double* __restrict__ velY,
            double* __restrict__ velZ,
            const double* __restrict__ accX,
            const double* __restrict__ accY,
            const double* __restrict__ accZ,
            const double* __restrict__ vacuumPressureRatio,
            const double* __restrict__ vibrationalDensity,
            double* __restrict__ localProperTime,
            std::size_t bodyCount,
            double dt,
            bool nonLinearTimeEnabled,
            double baseDensity
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

            // Advance each entity's local proper time clock using Essonne + Danaan coupling.
            const double timeRate = nonLinearTimeEnabled
                ? ComputeLocalTimeRate(vacuumPressureRatio[i], vibrationalDensity[i], baseDensity)
                : 1.0;
            localProperTime[i] += dt * timeRate;
        }

        /**
         * @brief Velocity Verlet stage 3: symmetric second half-step velocity kick.
         */
        __global__ void UnifiedFinalKickKernel(
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
         * @brief Optional 1st-order explicit Euler step for pedagogical comparisons.
         */
        __global__ void UnifiedEulerStepKernel(
            double* __restrict__ posX,
            double* __restrict__ posY,
            double* __restrict__ posZ,
            double* __restrict__ velX,
            double* __restrict__ velY,
            double* __restrict__ velZ,
            const double* __restrict__ accX,
            const double* __restrict__ accY,
            const double* __restrict__ accZ,
            const double* __restrict__ vacuumPressureRatio,
            const double* __restrict__ vibrationalDensity,
            double* __restrict__ localProperTime,
            std::size_t bodyCount,
            double dt,
            bool nonLinearTimeEnabled,
            double baseDensity
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

            const double timeRate = nonLinearTimeEnabled
                ? ComputeLocalTimeRate(vacuumPressureRatio[i], vibrationalDensity[i], baseDensity)
                : 1.0;
            localProperTime[i] += dt * timeRate;
        }

    } // anonymous namespace

    UnifiedFieldSolver::~UnifiedFieldSolver() {
        Shutdown();
    }

    bool UnifiedFieldSolver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        Shutdown();

        if (!context.IsInitialized()) {
            spdlog::error("UnifiedFieldSolver::Initialize failed: CUDAContext is not initialized.");
            return false;
        }

        std::string validationErr;
        if (!scene.Validate(validationErr)) {
            spdlog::error("UnifiedFieldSolver::Initialize failed: invalid scene ({}).", validationErr);
            return false;
        }

        if (scene.bodies.empty()) {
            spdlog::error("UnifiedFieldSolver::Initialize failed: scene '{}' contains 0 bodies.", scene.universeId);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(0U);
        m_bodyCount = scene.bodies.size();
        m_profileType = scene.physicsProfile.type;
        m_integrator = scene.physicsProfile.integrator;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_unifiedParams = scene.physicsProfile.unifiedField;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        // Allocate all 16 FP64 SoA GPU buffers under `CUDAContext`'s VRAM budget.
        if (!m_posX.Allocate(context, m_bodyCount) ||
            !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) ||
            !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) ||
            !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) ||
            !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) ||
            !m_mass.Allocate(context, m_bodyCount) ||
            !m_radius.Allocate(context, m_bodyCount) ||
            !m_vibrationalDensity.Allocate(context, m_bodyCount) ||
            !m_coherence.Allocate(context, m_bodyCount) ||
            !m_phryllCharge.Allocate(context, m_bodyCount) ||
            !m_vacuumPressureRatio.Allocate(context, m_bodyCount) ||
            !m_localProperTime.Allocate(context, m_bodyCount)) {
            spdlog::error("UnifiedFieldSolver: Failed allocating FP64 SoA GPU buffers for {} bodies.", m_bodyCount);
            Shutdown();
            return false;
        }

        m_hostPosX.resize(m_bodyCount);
        m_hostPosY.resize(m_bodyCount);
        m_hostPosZ.resize(m_bodyCount);
        m_hostVelX.resize(m_bodyCount);
        m_hostVelY.resize(m_bodyCount);
        m_hostVelZ.resize(m_bodyCount);
        m_hostMass.resize(m_bodyCount);
        m_hostRadius.resize(m_bodyCount);
        m_hostDensity.resize(m_bodyCount);
        m_hostCoherence.resize(m_bodyCount);
        m_hostPhryll.resize(m_bodyCount);
        m_hostPressureRatio.assign(m_bodyCount, 1.0);
        m_hostProperTime.assign(m_bodyCount, 0.0);

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const scene::BodyEntity& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x;
            m_hostPosY[i] = b.position.y;
            m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x;
            m_hostVelY[i] = b.velocity.y;
            m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;
            m_hostRadius[i] = b.radius;
            m_hostDensity[i] = b.vibrationalDensity;
            m_hostCoherence[i] = b.consciousnessCoherence;
            m_hostPhryll[i] = b.phryllCharge;
        }

        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) ||
            !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) ||
            !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) ||
            !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) ||
            !m_radius.CopyFromHost(m_hostRadius, m_stream) ||
            !m_vibrationalDensity.CopyFromHost(m_hostDensity, m_stream) ||
            !m_coherence.CopyFromHost(m_hostCoherence, m_stream) ||
            !m_phryllCharge.CopyFromHost(m_hostPhryll, m_stream) ||
            !m_accX.ZeroFill(m_stream) ||
            !m_accY.ZeroFill(m_stream) ||
            !m_accZ.ZeroFill(m_stream) ||
            !m_vacuumPressureRatio.CopyFromHost(m_hostPressureRatio, m_stream) ||
            !m_localProperTime.ZeroFill(m_stream)) {
            spdlog::error("UnifiedFieldSolver: Failed uploading initial unified state to GPU.");
            Shutdown();
            return false;
        }

        if (!LaunchUnifiedFieldKernel() || !Synchronize()) {
            spdlog::error("UnifiedFieldSolver: Initial unified field kernel failed.");
            Shutdown();
            return false;
        }

        m_initialized = true;

        SolverDiagnostics initialDiag{};
        if (!ComputeSolverDiagnostics(initialDiag)) {
            Shutdown();
            return false;
        }
        m_initialTotalEnergy = initialDiag.totalEnergy;

        spdlog::info(
            "UnifiedFieldSolver initialized for '{}' ({} bodies, aetherVortex={}, loveCoupling={:.4g}, nonLinearTime={}, E0={:.10g}).",
            scene.name,
            m_bodyCount,
            m_unifiedParams.aetherVortexEnabled ? "ON" : "OFF",
            m_unifiedParams.loveBindingCoupling,
            m_unifiedParams.nonLinearTimeEnabled ? "ON" : "OFF",
            m_initialTotalEnergy
        );
        return true;
    }

    void UnifiedFieldSolver::Shutdown() noexcept {
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
        m_radius.Release();
        m_vibrationalDensity.Release();
        m_coherence.Release();
        m_phryllCharge.Release();
        m_vacuumPressureRatio.Release();
        m_localProperTime.Release();

        m_context = nullptr;
        m_stream = nullptr;
        m_initialized = false;
        m_bodyCount = 0U;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_initialTotalEnergy = 0.0;
    }

    bool UnifiedFieldSolver::LaunchUnifiedFieldKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }

        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kFieldTileSize);
        if (gridSize == 0U) {
            return true;
        }

        const double softeningSq = m_softeningLength * m_softeningLength;
        ComputeUnifiedFieldKernel<kFieldTileSize><<<gridSize, kFieldTileSize, 0, m_stream>>>(
            m_posX.Data(),
            m_posY.Data(),
            m_posZ.Data(),
            m_mass.Data(),
            m_coherence.Data(),
            m_phryllCharge.Data(),
            m_accX.Data(),
            m_accY.Data(),
            m_accZ.Data(),
            m_vacuumPressureRatio.Data(),
            m_bodyCount,
            m_gravitationalConstant,
            softeningSq,
            m_unifiedParams.referenceLightSpeed,
            m_unifiedParams.aetherVortexEnabled,
            m_unifiedParams.aetherVortexCriticalRadius,
            m_unifiedParams.aetherVortexScaleHeight,
            m_unifiedParams.aetherVortexVerticalFieldGz,
            m_unifiedParams.loveBindingCoupling
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool UnifiedFieldSolver::Step(double dt) {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!std::isfinite(dt) || dt == 0.0) {
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
            UnifiedKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_vacuumPressureRatio.Data(),
                m_vibrationalDensity.Data(),
                m_localProperTime.Data(),
                m_bodyCount,
                dt,
                m_unifiedParams.nonLinearTimeEnabled,
                m_unifiedParams.baseVibrationalDensity
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }

            if (!LaunchUnifiedFieldKernel()) {
                return false;
            }

            UnifiedFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            UnifiedEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_vacuumPressureRatio.Data(),
                m_vibrationalDensity.Data(),
                m_localProperTime.Data(),
                m_bodyCount,
                dt,
                m_unifiedParams.nonLinearTimeEnabled,
                m_unifiedParams.baseVibrationalDensity
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }

            if (!LaunchUnifiedFieldKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool UnifiedFieldSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool UnifiedFieldSolver::Synchronize() const noexcept {
        if (m_context == nullptr || !m_context->IsInitialized()) {
            return false;
        }
        return m_context->SynchronizeStream(0U);
    }

    bool UnifiedFieldSolver::DownloadStateToStaging() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!m_posX.CopyToHost(m_hostPosX, m_stream) ||
            !m_posY.CopyToHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyToHost(m_hostPosZ, m_stream) ||
            !m_velX.CopyToHost(m_hostVelX, m_stream) ||
            !m_velY.CopyToHost(m_hostVelY, m_stream) ||
            !m_velZ.CopyToHost(m_hostVelZ, m_stream) ||
            !m_vacuumPressureRatio.CopyToHost(m_hostPressureRatio, m_stream) ||
            !m_localProperTime.CopyToHost(m_hostProperTime, m_stream)) {
            return false;
        }
        return Synchronize();
    }

    bool UnifiedFieldSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool UnifiedFieldSolver::GetLocalProperTimes(std::vector<double>& outProperTimes) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outProperTimes = m_hostProperTime;
        return true;
    }

    bool UnifiedFieldSolver::GetVacuumPressureRatios(std::vector<double>& outPressureRatios) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outPressureRatios = m_hostPressureRatio;
        return true;
    }

    bool UnifiedFieldSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0;
        double potential = 0.0;
        double totalMass = 0.0;
        scene::Vec3d linearMomentum{};
        scene::Vec3d angularMomentum{};
        scene::Vec3d weightedPos{};

        double sumPressureRatio = 0.0;
        double sumLightSpeedRatio = 0.0;
        double sumLocalTimeRate = 0.0;
        double sumProperTime = 0.0;
        double sumPhi = 0.0;
        double totalPhryll = m_unifiedParams.phryllFieldAmplitude;

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

            const double pRatio = m_hostPressureRatio[i];
            sumPressureRatio += pRatio;
            sumLightSpeedRatio += (pRatio > 0.0) ? std::sqrt(pRatio) : 0.0;
            sumLocalTimeRate += m_unifiedParams.nonLinearTimeEnabled
                ? ComputeLocalTimeRate(pRatio, m_hostDensity[i], m_unifiedParams.baseVibrationalDensity)
                : 1.0;
            sumProperTime += m_hostProperTime[i];
            sumPhi += ComputeHolographicRatioPhi(m_hostRadius[i]);
            totalPhryll += m_hostPhryll[i];

            // Accumulate conservative potential energy U matching the active force laws.
            if (m_unifiedParams.aetherVortexEnabled && i > 0U) {
                const double dx = ri.x - m_hostPosX[0];
                const double dy = ri.y - m_hostPosY[0];
                const double cylR = std::sqrt((dx * dx) + (dy * dy));
                potential += mi * ComputeEssonneAetherVortexPotential(
                    cylR,
                    m_hostMass[0],
                    m_gravitationalConstant,
                    m_unifiedParams.aetherVortexCriticalRadius,
                    m_unifiedParams.aetherVortexScaleHeight,
                    m_unifiedParams.aetherVortexVerticalFieldGz
                );
            }

            for (std::size_t j = i + 1U; j < m_bodyCount; ++j) {
                const double dx = m_hostPosX[j] - ri.x;
                const double dy = m_hostPosY[j] - ri.y;
                const double dz = m_hostPosZ[j] - ri.z;
                const double dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz) + softeningSq);
                if (dist > 0.0) {
                    if (!(m_unifiedParams.aetherVortexEnabled && i == 0U)) {
                        potential -= (m_gravitationalConstant * mi * m_hostMass[j]) / dist;
                    }
                    if (m_unifiedParams.loveBindingCoupling != 0.0) {
                        potential -= (m_unifiedParams.loveBindingCoupling *
                                      m_hostCoherence[i] * m_hostCoherence[j] *
                                      (m_hostPhryll[i] + m_hostPhryll[j])) / dist;
                    }
                }
            }
        }

        const double totalE = kinetic + potential;
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0)
            ? totalE
            : m_initialTotalEnergy;
        const double denom = std::max(std::abs(baselineE), 1.0e-30);
        const double invN = (m_bodyCount > 0U) ? (1.0 / static_cast<double>(m_bodyCount)) : 0.0;

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.kineticEnergy = kinetic;
        outDiagnostics.potentialEnergy = potential;
        outDiagnostics.totalEnergy = totalE;
        outDiagnostics.initialTotalEnergy = baselineE;
        outDiagnostics.relativeEnergyDrift = std::abs(totalE - baselineE) / denom;
        outDiagnostics.totalLinearMomentum = linearMomentum;
        outDiagnostics.totalAngularMomentum = angularMomentum;
        outDiagnostics.centerOfMass = (totalMass > 0.0) ? (weightedPos * (1.0 / totalMass)) : scene::Vec3d{};

        outDiagnostics.hasUnifiedTelemetry = true;
        outDiagnostics.meanVacuumPressureRatio = sumPressureRatio * invN;
        outDiagnostics.meanLocalLightSpeedRatio = sumLightSpeedRatio * invN;
        outDiagnostics.meanLocalTimeRate = sumLocalTimeRate * invN;
        outDiagnostics.meanLocalProperTime = sumProperTime * invN;
        outDiagnostics.meanHolographicPhi = sumPhi * invN;
        outDiagnostics.totalPhryllFlux = totalPhryll;
        outDiagnostics.solverSummary = m_unifiedParams.aetherVortexEnabled
            ? "Unified Field (Essonne Aether Vortex + Haramein + Love/Phryll)"
            : "Unified Field (Vacuum Gradient + Haramein + Love/Phryll)";

        return true;
    }

} // namespace umgebung::models::unified
