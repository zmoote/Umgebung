/**
 * @file HarameinSolver.cu
 * @brief FP64 CUDA implementation of the standalone `Thinker::Haramein` module
 *        (`Haramein342013PRRI3363.pdf` PSU holographic ratio $\phi$ +
 *        Haramein-Rauscher Coriolis spacetime torque via exact symplectic Strang splitting).
 */

#include "umgebung/models/thinkers/HarameinSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/thinkers/HarameinModel.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::thinkers {

    namespace {

        template <int kTileSize>
        __global__ void ComputeHarameinGravityKernel(
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
            __shared__ double sPosX[kTileSize];
            __shared__ double sPosY[kTileSize];
            __shared__ double sPosZ[kTileSize];
            __shared__ double sMass[kTileSize];

            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);

            double xi = 0.0, yi = 0.0, zi = 0.0;
            if (i < bodyCount) {
                xi = posX[i];
                yi = posY[i];
                zi = posZ[i];
            }

            double ax = 0.0, ay = 0.0, az = 0.0;

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

            if (i < bodyCount) {
                accX[i] = ax;
                accY[i] = ay;
                accZ[i] = az;
            }
        }

        /**
         * @brief Symplectic Strang-split Stage 1: half gravity kick, exact
         *        orthogonal Coriolis half-rotation (angle `omegaZ * dt`), full drift.
         *
         * @details
         * Because the Coriolis rotation $R(\Omega_z \Delta t)$ is strictly orthogonal
         * ($\det R = 1$, $\|R\mathbf{v}\| = \|\mathbf{v}\|$), it imparts Haramein-Rauscher
         * spacetime torque $\mathbf{a}_{\text{Cor}} = -2\boldsymbol{\Omega}\times\mathbf{v}$
         * while doing zero spurious work and remaining 2nd-order time-reversible!
         */
        __global__ void HarameinStrangKickDriftKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt,
            double omegaZ
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }

            const double halfDt = 0.5 * dt;
            double vx = velX[i] + (halfDt * accX[i]);
            double vy = velY[i] + (halfDt * accY[i]);
            const double vz = velZ[i] + (halfDt * accZ[i]);

            if (omegaZ != 0.0) {
                const double theta = omegaZ * dt;
                const double c = cos(theta);
                const double s = sin(theta);
                const double rvx = (vx * c) + (vy * s);
                const double rvy = (-vx * s) + (vy * c);
                vx = rvx;
                vy = rvy;
            }

            velX[i] = vx;
            velY[i] = vy;
            velZ[i] = vz;

            posX[i] += dt * vx;
            posY[i] += dt * vy;
            posZ[i] += dt * vz;
        }

        /**
         * @brief Symplectic Strang-split Stage 3: symmetric Coriolis half-rotation
         *        followed by second half gravity kick.
         */
        __global__ void HarameinStrangFinalKickKernel(
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt,
            double omegaZ
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }

            double vx = velX[i];
            double vy = velY[i];
            if (omegaZ != 0.0) {
                const double theta = omegaZ * dt;
                const double c = cos(theta);
                const double s = sin(theta);
                const double rvx = (vx * c) + (vy * s);
                const double rvy = (-vx * s) + (vy * c);
                vx = rvx;
                vy = rvy;
            }

            const double halfDt = 0.5 * dt;
            velX[i] = vx + (halfDt * accX[i]);
            velY[i] = vy + (halfDt * accY[i]);
            velZ[i] += halfDt * accZ[i];
        }

        __global__ void HarameinEulerStepKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt,
            double omegaZ
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }
            const double vx = velX[i];
            const double vy = velY[i];
            const double vz = velZ[i];
            posX[i] += dt * vx;
            posY[i] += dt * vy;
            posZ[i] += dt * vz;
            velX[i] = vx + dt * (accX[i] + ComputeHarameinCoriolisAccelX(vy, omegaZ));
            velY[i] = vy + dt * (accY[i] + ComputeHarameinCoriolisAccelY(vx, omegaZ));
            velZ[i] = vz + dt * accZ[i];
        }

    } // anonymous namespace

    HarameinSolver::~HarameinSolver() {
        Shutdown();
    }

    bool HarameinSolver::Initialize(compute::CUDAContext& context, const scene::UniverseScene& scene) {
        Shutdown();
        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err) || scene.bodies.empty()) {
            spdlog::error("HarameinSolver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_bodyCount = scene.bodies.size();
        m_integrator = scene.physicsProfile.integrator;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_harameinParams = scene.physicsProfile.unifiedField.haramein;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        if (!m_posX.Allocate(context, m_bodyCount) || !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) || !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) || !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) || !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) || !m_mass.Allocate(context, m_bodyCount)) {
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

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const auto& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x;
            m_hostPosY[i] = b.position.y;
            m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x;
            m_hostVelY[i] = b.velocity.y;
            m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;
            m_hostRadius[i] = b.radius;
        }

        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) || !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) || !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) || !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) || !m_accX.ZeroFill(m_stream) ||
            !m_accY.ZeroFill(m_stream) || !m_accZ.ZeroFill(m_stream)) {
            Shutdown();
            return false;
        }

        if (!LaunchGravityKernel() || !Synchronize()) {
            Shutdown();
            return false;
        }

        m_initialized = true;
        SolverDiagnostics initDiag{};
        if (!ComputeSolverDiagnostics(initDiag)) {
            Shutdown();
            return false;
        }
        m_initialTotalEnergy = initDiag.totalEnergy;
        return true;
    }

    void HarameinSolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_posX.Release(); m_posY.Release(); m_posZ.Release();
        m_velX.Release(); m_velY.Release(); m_velZ.Release();
        m_accX.Release(); m_accY.Release(); m_accZ.Release();
        m_mass.Release();
        m_context = nullptr; m_stream = nullptr; m_initialized = false;
        m_bodyCount = 0U; m_stepCount = 0ULL; m_simulationTime = 0.0; m_initialTotalEnergy = 0.0;
    }

    bool HarameinSolver::LaunchGravityKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }
        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kTileSize);
        if (gridSize == 0U) {
            return true;
        }
        ComputeHarameinGravityKernel<kTileSize><<<gridSize, kTileSize, 0, m_stream>>>(
            m_posX.Data(), m_posY.Data(), m_posZ.Data(), m_mass.Data(),
            m_accX.Data(), m_accY.Data(), m_accZ.Data(),
            m_bodyCount, m_gravitationalConstant, m_softeningLength * m_softeningLength
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool HarameinSolver::Step(double dt) {
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
            HarameinStrangKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount, dt, m_harameinParams.spacetimeTorqueOmegaZ
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchGravityKernel()) {
                return false;
            }
            HarameinStrangFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount, dt, m_harameinParams.spacetimeTorqueOmegaZ
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            HarameinEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount, dt, m_harameinParams.spacetimeTorqueOmegaZ
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchGravityKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool HarameinSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool HarameinSolver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool HarameinSolver::DownloadStateToStaging() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!m_posX.CopyToHost(m_hostPosX, m_stream) || !m_posY.CopyToHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyToHost(m_hostPosZ, m_stream) || !m_velX.CopyToHost(m_hostVelX, m_stream) ||
            !m_velY.CopyToHost(m_hostVelY, m_stream) || !m_velZ.CopyToHost(m_hostVelZ, m_stream)) {
            return false;
        }
        return Synchronize();
    }

    bool HarameinSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool HarameinSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0, potential = 0.0, totalMass = 0.0, sumPhi = 0.0;
        scene::Vec3d linearMomentum{}, angularMomentum{}, weightedPos{};
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
            sumPhi += ComputeHolographicRatioPhi(m_hostRadius[i]);

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
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0) ? totalE : m_initialTotalEnergy;
        const double invN = (m_bodyCount > 0U) ? (1.0 / static_cast<double>(m_bodyCount)) : 0.0;

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::ThinkerHaramein;
        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.kineticEnergy = kinetic;
        outDiagnostics.potentialEnergy = potential;
        outDiagnostics.totalEnergy = totalE;
        outDiagnostics.initialTotalEnergy = baselineE;
        outDiagnostics.relativeEnergyDrift = std::abs(totalE - baselineE) / std::max(std::abs(baselineE), 1.0e-30);
        outDiagnostics.totalLinearMomentum = linearMomentum;
        outDiagnostics.totalAngularMomentum = angularMomentum;
        outDiagnostics.centerOfMass = (totalMass > 0.0) ? (weightedPos * (1.0 / totalMass)) : scene::Vec3d{};
        outDiagnostics.hasUnifiedTelemetry = true;
        outDiagnostics.meanLocalProperTime = m_simulationTime;
        outDiagnostics.meanHolographicPhi = sumPhi * invN;
        outDiagnostics.harameinSpacetimeTorqueOmegaZ = m_harameinParams.spacetimeTorqueOmegaZ;
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::thinkers
