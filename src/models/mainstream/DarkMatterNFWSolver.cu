/**
 * @file DarkMatterNFWSolver.cu
 * @brief FP64 CUDA implementation of Pillar 1's `Mainstream::DarkMatterNFW`
 *        baseline solver (pairwise Newtonian gravity + spherical Navarro-Frenk-White
 *        Cold Dark Matter halo potential with symplectic Velocity Verlet).
 */

#include "umgebung/models/mainstream/DarkMatterNFWSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::mainstream {

    namespace {

        /**
         * @brief Shared-memory tiled FP64 pairwise Newtonian gravity + static
         *        spherical NFW dark matter halo acceleration kernel.
         *
         * @details
         * For each body $i \in [0, N)$:
         *   1. Sums exact pairwise Newtonian gravitational acceleration over all $j \neq i$.
         *   2. Adds the conservative inward spherical NFW halo acceleration
         *      $\mathbf{a}_i^{\text{NFW}} = -g_{\text{NFW}}(r_i)\,\hat{\mathbf{r}}_i$
         *      relative to the fixed halo center `(haloCx, haloCy, haloCz)`.
         */
        template <int kTileSize>
        __global__ void ComputeNFWAccelerationsTiledKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
            double* __restrict__ accX,
            double* __restrict__ accY,
            double* __restrict__ accZ,
            std::size_t bodyCount,
            double G,
            double softeningSq,
            double nfwRho0,
            double nfwRs,
            double haloCx,
            double haloCy,
            double haloCz
        ) {
            __shared__ double sPosX[kTileSize];
            __shared__ double sPosY[kTileSize];
            __shared__ double sPosZ[kTileSize];
            __shared__ double sMass[kTileSize];

            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);

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
                            const double invDist = 1.0 / sqrt(distSq);
                            const double invDist3 = invDist * invDist * invDist;
                            const double factor = G * sMass[k] * invDist3;
                            ax += dx * factor;
                            ay += dy * factor;
                            az += dz * factor;
                        }
                    }
                }

                __syncthreads();
            }

            // Add spherical NFW halo inward gravitational acceleration on orbiting
            // bodies (i > 0) toward the central halo core `m_haloCenter` (avoids
            // applying the r -> 0 NFW cusp force to the central galactic core itself).
            if (i > 0U && i < bodyCount && nfwRho0 > 0.0 && nfwRs > 0.0) {
                const double rx = xi - haloCx;
                const double ry = yi - haloCy;
                const double rz = zi - haloCz;
                const double rSq = (rx * rx) + (ry * ry) + (rz * rz);
                if (rSq > 0.0) {
                    const double r = sqrt(rSq);
                    const double gNfw = ComputeNFWRadialAccel(r, G, nfwRho0, nfwRs);
                    const double invR = 1.0 / r;
                    ax -= gNfw * (rx * invR);
                    ay -= gNfw * (ry * invR);
                    az -= gNfw * (rz * invR);
                }
            }

            if (i < bodyCount) {
                accX[i] = ax;
                accY[i] = ay;
                accZ[i] = az;
            }
        }

        __global__ void NFVKickDriftKernel(
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

        __global__ void NFWFinalKickKernel(
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

        __global__ void NFWEulerStepKernel(
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

    DarkMatterNFWSolver::~DarkMatterNFWSolver() {
        Shutdown();
    }

    bool DarkMatterNFWSolver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        Shutdown();

        if (!context.IsInitialized()) {
            spdlog::error("DarkMatterNFWSolver::Initialize failed: CUDAContext is not initialized.");
            return false;
        }

        std::string validationErr;
        if (!scene.Validate(validationErr) || scene.bodies.empty()) {
            spdlog::error("DarkMatterNFWSolver::Initialize failed: invalid or empty scene ({}).", validationErr);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_bodyCount = scene.bodies.size();
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_nfwParams = scene.physicsProfile.darkMatterNFW;
        m_haloCenter = scene.bodies[0].position;
        m_integrator = scene.physicsProfile.integrator;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

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
            spdlog::error("DarkMatterNFWSolver: Failed allocating FP64 SoA GPU buffers.");
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
            spdlog::error("DarkMatterNFWSolver: Failed uploading initial scene state to GPU.");
            Shutdown();
            return false;
        }

        if (!LaunchNFWAccelerationKernel() || !Synchronize()) {
            spdlog::error("DarkMatterNFWSolver: Initial NFW acceleration kernel failed.");
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
            "DarkMatterNFWSolver initialized for '{}' ({} bodies, rho0={:.6g}, r_s={:.4g}, E0={:.10g}).",
            scene.name,
            m_bodyCount,
            m_nfwParams.characteristicDensityRho0,
            m_nfwParams.scaleRadiusRs,
            m_initialTotalEnergy
        );
        return true;
    }

    void DarkMatterNFWSolver::Shutdown() noexcept {
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

        m_context = nullptr;
        m_stream = nullptr;
        m_initialized = false;
        m_bodyCount = 0U;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_initialTotalEnergy = 0.0;
    }

    bool DarkMatterNFWSolver::LaunchNFWAccelerationKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }

        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kGravityTileSize);
        if (gridSize == 0U) {
            return true;
        }

        const double softeningSq = m_softeningLength * m_softeningLength;
        ComputeNFWAccelerationsTiledKernel<kGravityTileSize><<<gridSize, kGravityTileSize, 0, m_stream>>>(
            m_posX.Data(), m_posY.Data(), m_posZ.Data(), m_mass.Data(),
            m_accX.Data(), m_accY.Data(), m_accZ.Data(),
            m_bodyCount,
            m_gravitationalConstant,
            softeningSq,
            m_nfwParams.characteristicDensityRho0,
            m_nfwParams.scaleRadiusRs,
            m_haloCenter.x,
            m_haloCenter.y,
            m_haloCenter.z
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool DarkMatterNFWSolver::Step(double dt) {
        if (!m_initialized || m_context == nullptr || !std::isfinite(dt) || dt == 0.0) {
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
            NFVKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchNFWAccelerationKernel()) {
                return false;
            }
            NFWFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            NFWEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount,
                dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchNFWAccelerationKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool DarkMatterNFWSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool DarkMatterNFWSolver::Synchronize() const noexcept {
        if (m_context == nullptr || !m_context->IsInitialized()) {
            return false;
        }
        return m_context->SynchronizeStream(m_streamIndex);
    }

    bool DarkMatterNFWSolver::DownloadStateToStaging() {
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

    bool DarkMatterNFWSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool DarkMatterNFWSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0;
        double baryonPotential = 0.0;
        double nfwPotential = 0.0;
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

            // Accumulate conservative NFW halo potential energy U_NFW = sum_{i>0} m_i Phi_NFW(|r_i - r_halo|).
            if (i > 0U) {
                const scene::Vec3d rel = ri - m_haloCenter;
                nfwPotential += mi * ComputeNFWPotential(
                    rel.Norm(),
                    m_gravitationalConstant,
                    m_nfwParams.characteristicDensityRho0,
                    m_nfwParams.scaleRadiusRs
                );
            }

            for (std::size_t j = i + 1U; j < m_bodyCount; ++j) {
                const double dx = m_hostPosX[j] - ri.x;
                const double dy = m_hostPosY[j] - ri.y;
                const double dz = m_hostPosZ[j] - ri.z;
                const double dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz) + softeningSq);
                if (dist > 0.0) {
                    baryonPotential -= (m_gravitationalConstant * mi * m_hostMass[j]) / dist;
                }
            }
        }

        const double totalPotential = baryonPotential + nfwPotential;
        const double totalE = kinetic + totalPotential;
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0)
            ? totalE
            : m_initialTotalEnergy;
        const double denom = std::max(std::abs(baselineE), 1.0e-30);

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::MainstreamDarkMatterNFW;
        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.kineticEnergy = kinetic;
        outDiagnostics.potentialEnergy = totalPotential;
        outDiagnostics.nfwHaloPotentialEnergy = nfwPotential;
        outDiagnostics.totalEnergy = totalE;
        outDiagnostics.initialTotalEnergy = baselineE;
        outDiagnostics.relativeEnergyDrift = std::abs(totalE - baselineE) / denom;
        outDiagnostics.totalLinearMomentum = linearMomentum;
        outDiagnostics.totalAngularMomentum = angularMomentum;
        outDiagnostics.centerOfMass = (totalMass > 0.0) ? (weightedPos * (1.0 / totalMass)) : scene::Vec3d{};
        outDiagnostics.meanLocalProperTime = m_simulationTime;
        outDiagnostics.hasUnifiedTelemetry = false;
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::mainstream
