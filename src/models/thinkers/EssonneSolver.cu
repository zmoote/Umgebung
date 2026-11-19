/**
 * @file EssonneSolver.cu
 * @brief FP64 CUDA implementation of the standalone `Thinker::Essonne` module
 *        (`8052.pdf` vacuum energy density gradient + `10100.pdf` 2D spiral-galaxy
 *        aether vortex).
 */

#include "umgebung/models/thinkers/EssonneSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/thinkers/EssonneModel.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::thinkers {

    namespace {

        template <int kTileSize>
        __global__ void ComputeEssonneFieldKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
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
            double vortexGz
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
            double pressureDepletion = 0.0;
            const double invC2 = (refLightSpeed > 0.0) ? (1.0 / (refLightSpeed * refLightSpeed)) : 0.0;

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
                            const double dist = sqrt(distSq);
                            const double invDist = 1.0 / dist;
                            const double invDist3 = invDist * invDist * invDist;

                            // Essonne vacuum pressure depletion 2 G m_j / (c^2 r_ij) (8052.pdf Eq. 21).
                            pressureDepletion += (2.0 * G * sMass[k] * invC2) * invDist;

                            if (aetherVortexEnabled && j == 0U && i > 0U) {
                                // Essonne 2D galactic aether-vortex radial field (10100.pdf Eq. 7 & 11).
                                const double cylR = sqrt((dx * dx) + (dy * dy));
                                if (cylR > 0.0) {
                                    const double gr = ComputeEssonneAetherVortexRadialAccel(
                                        cylR, sMass[k], G, vortexRc, vortexScaleH, vortexGz
                                    );
                                    const double invCylR = 1.0 / cylR;
                                    ax += gr * (dx * invCylR);
                                    ay += gr * (dy * invCylR);
                                }
                            } else if (!(aetherVortexEnabled && i == 0U)) {
                                const double gravScale = G * sMass[k] * invDist3;
                                ax += dx * gravScale;
                                ay += dy * gravScale;
                                az += dz * gravScale;
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

        __global__ void EssonneKickDriftKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            const double* __restrict__ vacuumPressureRatio,
            double* __restrict__ localProperTime,
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

            // Gravitational vacuum-pressure clock slowing dtau/dt = c'/c = sqrt(P/P0) (8052.pdf Eq. 27).
            const double ratio = vacuumPressureRatio[i];
            const double rate = (ratio > 0.0) ? sqrt(ratio) : 0.0;
            localProperTime[i] += dt * rate;
        }

        __global__ void EssonneFinalKickKernel(
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
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

        __global__ void EssonneEulerStepKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            const double* __restrict__ vacuumPressureRatio,
            double* __restrict__ localProperTime,
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
            const double ratio = vacuumPressureRatio[i];
            localProperTime[i] += dt * ((ratio > 0.0) ? sqrt(ratio) : 0.0);
        }

    } // anonymous namespace

    EssonneSolver::~EssonneSolver() {
        Shutdown();
    }

    bool EssonneSolver::Initialize(compute::CUDAContext& context, const scene::UniverseScene& scene) {
        Shutdown();
        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err) || scene.bodies.empty()) {
            spdlog::error("EssonneSolver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_bodyCount = scene.bodies.size();
        m_integrator = scene.physicsProfile.integrator;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_refLightSpeed = scene.physicsProfile.unifiedField.referenceLightSpeed;
        m_aetherVortexEnabled = scene.physicsProfile.unifiedField.aetherVortexEnabled;
        m_vortexRc = scene.physicsProfile.unifiedField.aetherVortexCriticalRadius;
        m_vortexH = scene.physicsProfile.unifiedField.aetherVortexScaleHeight;
        m_vortexGz = scene.physicsProfile.unifiedField.aetherVortexVerticalFieldGz;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        if (!m_posX.Allocate(context, m_bodyCount) || !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) || !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) || !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) || !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) || !m_mass.Allocate(context, m_bodyCount) ||
            !m_vacuumPressureRatio.Allocate(context, m_bodyCount) ||
            !m_localProperTime.Allocate(context, m_bodyCount)) {
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
        m_hostPressureRatio.assign(m_bodyCount, 1.0);
        m_hostProperTime.assign(m_bodyCount, 0.0);

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const auto& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x;
            m_hostPosY[i] = b.position.y;
            m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x;
            m_hostVelY[i] = b.velocity.y;
            m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;
        }

        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) || !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) || !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) || !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) || !m_accX.ZeroFill(m_stream) ||
            !m_accY.ZeroFill(m_stream) || !m_accZ.ZeroFill(m_stream) ||
            !m_vacuumPressureRatio.CopyFromHost(m_hostPressureRatio, m_stream) ||
            !m_localProperTime.ZeroFill(m_stream)) {
            Shutdown();
            return false;
        }

        if (!LaunchEssonneKernel() || !Synchronize()) {
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

    void EssonneSolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_posX.Release(); m_posY.Release(); m_posZ.Release();
        m_velX.Release(); m_velY.Release(); m_velZ.Release();
        m_accX.Release(); m_accY.Release(); m_accZ.Release();
        m_mass.Release(); m_vacuumPressureRatio.Release(); m_localProperTime.Release();
        m_context = nullptr; m_stream = nullptr; m_initialized = false;
        m_bodyCount = 0U; m_stepCount = 0ULL; m_simulationTime = 0.0; m_initialTotalEnergy = 0.0;
    }

    bool EssonneSolver::LaunchEssonneKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }
        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kTileSize);
        if (gridSize == 0U) {
            return true;
        }
        ComputeEssonneFieldKernel<kTileSize><<<gridSize, kTileSize, 0, m_stream>>>(
            m_posX.Data(), m_posY.Data(), m_posZ.Data(), m_mass.Data(),
            m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_vacuumPressureRatio.Data(),
            m_bodyCount, m_gravitationalConstant, m_softeningLength * m_softeningLength,
            m_refLightSpeed, m_aetherVortexEnabled, m_vortexRc, m_vortexH, m_vortexGz
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool EssonneSolver::Step(double dt) {
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
            EssonneKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_vacuumPressureRatio.Data(), m_localProperTime.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchEssonneKernel()) {
                return false;
            }
            EssonneFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            EssonneEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_vacuumPressureRatio.Data(), m_localProperTime.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchEssonneKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool EssonneSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool EssonneSolver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool EssonneSolver::DownloadStateToStaging() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!m_posX.CopyToHost(m_hostPosX, m_stream) || !m_posY.CopyToHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyToHost(m_hostPosZ, m_stream) || !m_velX.CopyToHost(m_hostVelX, m_stream) ||
            !m_velY.CopyToHost(m_hostVelY, m_stream) || !m_velZ.CopyToHost(m_hostVelZ, m_stream) ||
            !m_vacuumPressureRatio.CopyToHost(m_hostPressureRatio, m_stream) ||
            !m_localProperTime.CopyToHost(m_hostProperTime, m_stream)) {
            return false;
        }
        return Synchronize();
    }

    bool EssonneSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool EssonneSolver::GetLocalProperTimes(std::vector<double>& outProperTimes) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outProperTimes = m_hostProperTime;
        return true;
    }

    bool EssonneSolver::GetVacuumPressureRatios(std::vector<double>& outPressureRatios) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outPressureRatios = m_hostPressureRatio;
        return true;
    }

    bool EssonneSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0, potential = 0.0, totalMass = 0.0;
        scene::Vec3d linearMomentum{}, angularMomentum{}, weightedPos{};
        double sumPressure = 0.0, sumLight = 0.0, sumProperTime = 0.0;
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
            sumPressure += pRatio;
            sumLight += (pRatio > 0.0) ? std::sqrt(pRatio) : 0.0;
            sumProperTime += m_hostProperTime[i];

            if (m_aetherVortexEnabled && i > 0U) {
                const double dx = ri.x - m_hostPosX[0];
                const double dy = ri.y - m_hostPosY[0];
                potential += mi * ComputeEssonneAetherVortexPotential(
                    std::sqrt((dx * dx) + (dy * dy)),
                    m_hostMass[0], m_gravitationalConstant, m_vortexRc, m_vortexH, m_vortexGz
                );
            }

            for (std::size_t j = i + 1U; j < m_bodyCount; ++j) {
                const double dx = m_hostPosX[j] - ri.x;
                const double dy = m_hostPosY[j] - ri.y;
                const double dz = m_hostPosZ[j] - ri.z;
                const double dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz) + softeningSq);
                if (dist > 0.0 && !(m_aetherVortexEnabled && i == 0U)) {
                    potential -= (m_gravitationalConstant * mi * m_hostMass[j]) / dist;
                }
            }
        }

        const double totalE = kinetic + potential;
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0) ? totalE : m_initialTotalEnergy;
        const double invN = (m_bodyCount > 0U) ? (1.0 / static_cast<double>(m_bodyCount)) : 0.0;

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::ThinkerEssonne;
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
        outDiagnostics.meanVacuumPressureRatio = sumPressure * invN;
        outDiagnostics.meanLocalLightSpeedRatio = sumLight * invN;
        outDiagnostics.meanLocalTimeRate = sumLight * invN;
        outDiagnostics.meanLocalProperTime = sumProperTime * invN;
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::thinkers
