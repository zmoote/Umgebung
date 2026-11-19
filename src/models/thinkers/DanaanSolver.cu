/**
 * @file DanaanSolver.cu
 * @brief FP64 CUDA implementation of the standalone `Thinker::Danaan` module
 *        (Elena Danaan Love binding coherence force + vibrational density octaves
 *        + gravity-bound non-linear local proper time).
 */

#include "umgebung/models/thinkers/DanaanSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/thinkers/DanaanModel.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::thinkers {

    namespace {

        template <int kTileSize>
        __global__ void ComputeDanaanFieldKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
            const double* __restrict__ coherence,
            const double* __restrict__ phryll,
            double* __restrict__ accX,
            double* __restrict__ accY,
            double* __restrict__ accZ,
            double* __restrict__ gravTimeFactor,
            std::size_t bodyCount,
            double G,
            double softeningSq,
            double refLightSpeed,
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

            double xi = 0.0, yi = 0.0, zi = 0.0, mi = 1.0, ci = 0.0, qi = 0.0;
            if (i < bodyCount) {
                xi = posX[i]; yi = posY[i]; zi = posZ[i];
                mi = mass[i]; ci = coherence[i]; qi = phryll[i];
            }

            double ax = 0.0, ay = 0.0, az = 0.0;
            double gravDepletion = 0.0;
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
                    sPosX[threadIdx.x] = 0.0; sPosY[threadIdx.x] = 0.0; sPosZ[threadIdx.x] = 0.0;
                    sMass[threadIdx.x] = 0.0; sCoh[threadIdx.x] = 0.0; sPhryll[threadIdx.x] = 0.0;
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

                            gravDepletion += (2.0 * G * sMass[k] * invC2) * invDist;

                            const double gravScale = G * sMass[k] * invDist3;
                            ax += dx * gravScale;
                            ay += dy * gravScale;
                            az += dz * gravScale;

                            // Antisymmetric Love binding coherence force preserving linear momentum.
                            if (loveCoupling != 0.0 && mi > 0.0) {
                                const double loveForce = ComputeDanaanLoveForceScale(
                                    loveCoupling, ci, sCoh[k], qi, sPhryll[k], invDist3
                                );
                                const double loveAccel = loveForce / mi;
                                ax += dx * loveAccel;
                                ay += dy * loveAccel;
                                az += dz * loveAccel;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            if (i < bodyCount) {
                accX[i] = ax; accY[i] = ay; accZ[i] = az;
                const double ratio = 1.0 - gravDepletion;
                gravTimeFactor[i] = (ratio > 0.0) ? ratio : 0.0;
            }
        }

        __global__ void DanaanKickDriftKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            const double* __restrict__ gravTimeFactor,
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
            velX[i] = vxHalf; velY[i] = vyHalf; velZ[i] = vzHalf;
            posX[i] += dt * vxHalf; posY[i] += dt * vyHalf; posZ[i] += dt * vzHalf;

            const double rate = nonLinearTimeEnabled
                ? ComputeLocalTimeRate(gravTimeFactor[i], vibrationalDensity[i], baseDensity)
                : 1.0;
            localProperTime[i] += dt * rate;
        }

        __global__ void DanaanFinalKickKernel(
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            std::size_t bodyCount, double dt
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

        __global__ void DanaanEulerStepKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            const double* __restrict__ gravTimeFactor,
            const double* __restrict__ vibrationalDensity,
            double* __restrict__ localProperTime,
            std::size_t bodyCount, double dt, bool nonLinearTimeEnabled, double baseDensity
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }
            posX[i] += dt * velX[i]; posY[i] += dt * velY[i]; posZ[i] += dt * velZ[i];
            velX[i] += dt * accX[i]; velY[i] += dt * accY[i]; velZ[i] += dt * accZ[i];
            const double rate = nonLinearTimeEnabled
                ? ComputeLocalTimeRate(gravTimeFactor[i], vibrationalDensity[i], baseDensity)
                : 1.0;
            localProperTime[i] += dt * rate;
        }

    } // anonymous namespace

    DanaanSolver::~DanaanSolver() {
        Shutdown();
    }

    bool DanaanSolver::Initialize(compute::CUDAContext& context, const scene::UniverseScene& scene) {
        Shutdown();
        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err) || scene.bodies.empty()) {
            spdlog::error("DanaanSolver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_bodyCount = scene.bodies.size();
        m_integrator = scene.physicsProfile.integrator;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_refLightSpeed = scene.physicsProfile.unifiedField.referenceLightSpeed;
        m_loveBindingCoupling = scene.physicsProfile.unifiedField.loveBindingCoupling;
        m_baseVibrationalDensity = scene.physicsProfile.unifiedField.baseVibrationalDensity;
        m_phryllAmplitude = scene.physicsProfile.unifiedField.phryllFieldAmplitude;
        m_nonLinearTimeEnabled = scene.physicsProfile.unifiedField.nonLinearTimeEnabled;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        if (!m_posX.Allocate(context, m_bodyCount) || !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) || !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) || !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) || !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) || !m_mass.Allocate(context, m_bodyCount) ||
            !m_vibrationalDensity.Allocate(context, m_bodyCount) ||
            !m_coherence.Allocate(context, m_bodyCount) ||
            !m_phryllCharge.Allocate(context, m_bodyCount) ||
            !m_gravTimeFactor.Allocate(context, m_bodyCount) ||
            !m_localProperTime.Allocate(context, m_bodyCount)) {
            Shutdown();
            return false;
        }

        m_hostPosX.resize(m_bodyCount); m_hostPosY.resize(m_bodyCount); m_hostPosZ.resize(m_bodyCount);
        m_hostVelX.resize(m_bodyCount); m_hostVelY.resize(m_bodyCount); m_hostVelZ.resize(m_bodyCount);
        m_hostMass.resize(m_bodyCount); m_hostDensity.resize(m_bodyCount);
        m_hostCoherence.resize(m_bodyCount); m_hostPhryll.resize(m_bodyCount);
        m_hostGravTimeFactor.assign(m_bodyCount, 1.0);
        m_hostProperTime.assign(m_bodyCount, 0.0);

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const auto& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x; m_hostPosY[i] = b.position.y; m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x; m_hostVelY[i] = b.velocity.y; m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;
            m_hostDensity[i] = b.vibrationalDensity;
            m_hostCoherence[i] = b.consciousnessCoherence;
            m_hostPhryll[i] = b.phryllCharge;
        }

        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) || !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) || !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) || !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) ||
            !m_vibrationalDensity.CopyFromHost(m_hostDensity, m_stream) ||
            !m_coherence.CopyFromHost(m_hostCoherence, m_stream) ||
            !m_phryllCharge.CopyFromHost(m_hostPhryll, m_stream) ||
            !m_accX.ZeroFill(m_stream) || !m_accY.ZeroFill(m_stream) || !m_accZ.ZeroFill(m_stream) ||
            !m_gravTimeFactor.CopyFromHost(m_hostGravTimeFactor, m_stream) ||
            !m_localProperTime.ZeroFill(m_stream)) {
            Shutdown();
            return false;
        }

        if (!LaunchDanaanKernel() || !Synchronize()) {
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

    void DanaanSolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_posX.Release(); m_posY.Release(); m_posZ.Release();
        m_velX.Release(); m_velY.Release(); m_velZ.Release();
        m_accX.Release(); m_accY.Release(); m_accZ.Release();
        m_mass.Release(); m_vibrationalDensity.Release(); m_coherence.Release();
        m_phryllCharge.Release(); m_gravTimeFactor.Release(); m_localProperTime.Release();
        m_context = nullptr; m_stream = nullptr; m_initialized = false;
        m_bodyCount = 0U; m_stepCount = 0ULL; m_simulationTime = 0.0; m_initialTotalEnergy = 0.0;
    }

    bool DanaanSolver::LaunchDanaanKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }
        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kTileSize);
        if (gridSize == 0U) {
            return true;
        }
        ComputeDanaanFieldKernel<kTileSize><<<gridSize, kTileSize, 0, m_stream>>>(
            m_posX.Data(), m_posY.Data(), m_posZ.Data(), m_mass.Data(),
            m_coherence.Data(), m_phryllCharge.Data(),
            m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_gravTimeFactor.Data(),
            m_bodyCount, m_gravitationalConstant, m_softeningLength * m_softeningLength,
            m_refLightSpeed, m_loveBindingCoupling
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool DanaanSolver::Step(double dt) {
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
            DanaanKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_gravTimeFactor.Data(), m_vibrationalDensity.Data(), m_localProperTime.Data(),
                m_bodyCount, dt, m_nonLinearTimeEnabled, m_baseVibrationalDensity
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchDanaanKernel()) {
                return false;
            }
            DanaanFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            DanaanEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_gravTimeFactor.Data(), m_vibrationalDensity.Data(), m_localProperTime.Data(),
                m_bodyCount, dt, m_nonLinearTimeEnabled, m_baseVibrationalDensity
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchDanaanKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool DanaanSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool DanaanSolver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool DanaanSolver::DownloadStateToStaging() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!m_posX.CopyToHost(m_hostPosX, m_stream) || !m_posY.CopyToHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyToHost(m_hostPosZ, m_stream) || !m_velX.CopyToHost(m_hostVelX, m_stream) ||
            !m_velY.CopyToHost(m_hostVelY, m_stream) || !m_velZ.CopyToHost(m_hostVelZ, m_stream) ||
            !m_gravTimeFactor.CopyToHost(m_hostGravTimeFactor, m_stream) ||
            !m_localProperTime.CopyToHost(m_hostProperTime, m_stream)) {
            return false;
        }
        return Synchronize();
    }

    bool DanaanSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool DanaanSolver::GetLocalProperTimes(std::vector<double>& outProperTimes) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outProperTimes = m_hostProperTime;
        return true;
    }

    bool DanaanSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0, potential = 0.0, totalMass = 0.0;
        scene::Vec3d linearMomentum{}, angularMomentum{}, weightedPos{};
        double sumRate = 0.0, sumProper = 0.0, totalPhryll = m_phryllAmplitude;
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

            sumRate += m_nonLinearTimeEnabled
                ? ComputeLocalTimeRate(m_hostGravTimeFactor[i], m_hostDensity[i], m_baseVibrationalDensity)
                : 1.0;
            sumProper += m_hostProperTime[i];
            totalPhryll += m_hostPhryll[i];

            for (std::size_t j = i + 1U; j < m_bodyCount; ++j) {
                const double dx = m_hostPosX[j] - ri.x;
                const double dy = m_hostPosY[j] - ri.y;
                const double dz = m_hostPosZ[j] - ri.z;
                const double dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz) + softeningSq);
                if (dist > 0.0) {
                    potential -= (m_gravitationalConstant * mi * m_hostMass[j]) / dist;
                    if (m_loveBindingCoupling != 0.0) {
                        potential -= (m_loveBindingCoupling * m_hostCoherence[i] * m_hostCoherence[j] *
                                      (m_hostPhryll[i] + m_hostPhryll[j])) / dist;
                    }
                }
            }
        }

        const double totalE = kinetic + potential;
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0) ? totalE : m_initialTotalEnergy;
        const double invN = (m_bodyCount > 0U) ? (1.0 / static_cast<double>(m_bodyCount)) : 0.0;

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::ThinkerDanaan;
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
        outDiagnostics.meanLocalTimeRate = sumRate * invN;
        outDiagnostics.meanLocalProperTime = sumProper * invN;
        outDiagnostics.totalPhryllFlux = totalPhryll;
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::thinkers
