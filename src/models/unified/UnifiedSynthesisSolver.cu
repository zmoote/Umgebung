/**
 * @file UnifiedSynthesisSolver.cu
 * @brief FP64 CUDA implementation of Tier 3 `Unified::Synthesis`, composing all
 *        five standalone Thinker equation modules (`EssonneModel`, `HarameinModel`,
 *        `PaisModel`, `DanaanModel`, `VogelWillisModel`) into a single GPU pipeline.
 */

#include "umgebung/models/unified/UnifiedSynthesisSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/unified/UnifiedPhysicsEquations.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::unified {

    namespace {

        template <int kTileSize>
        __global__ void ComputeUnifiedSynthesisKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
            const double* __restrict__ coherence,
            const double* __restrict__ phryll,
            const double* __restrict__ accelScale,
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
            double loveCoupling,
            double vogelPhryllCoupling,
            double vogelDrivePower
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

            double xi = 0.0, yi = 0.0, zi = 0.0, mi = 1.0, ci = 0.0, qi = 0.0, scaleI = 1.0;
            if (i < bodyCount) {
                xi = posX[i]; yi = posY[i]; zi = posZ[i];
                mi = mass[i]; ci = coherence[i]; qi = phryll[i]; scaleI = accelScale[i];
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

                            // 1. Essonne vacuum pressure depletion (8052.pdf Eq. 21).
                            pressureDepletion += (2.0 * G * sMass[k] * invC2) * invDist;

                            // 2. Essonne 2D galactic aether-vortex (10100.pdf Eq. 7/11) or pairwise gravity.
                            if (aetherVortexEnabled && j == 0U && i > 0U) {
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

                            // 3. Elena Danaan antisymmetric Love binding coherence force.
                            if (loveCoupling != 0.0 && mi > 0.0) {
                                const double loveF = ComputeDanaanLoveForceScale(
                                    loveCoupling, ci, sCoh[k], qi, sPhryll[k], invDist3
                                );
                                const double loveA = loveF / mi;
                                ax += dx * loveA;
                                ay += dy * loveA;
                                az += dz * loveA;
                            }

                            // 4. Vogel/Willis 4096 Hz torsion-coil Phryll crystal resonant coupling.
                            if (vogelPhryllCoupling != 0.0 && vogelDrivePower > 0.0 && mi > 0.0) {
                                const double phryllF = ComputeVogelPhryllForceScale(
                                    vogelPhryllCoupling, vogelDrivePower, qi, sPhryll[k], invDist3
                                );
                                const double phryllA = phryllF / mi;
                                ax += dx * phryllA;
                                ay += dy * phryllA;
                                az += dz * phryllA;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            if (i < bodyCount) {
                // 5. Salvatore Pais HEEMFG inertial mass reduction (m_0 / m_eff amplification).
                accX[i] = ax * scaleI;
                accY[i] = ay * scaleI;
                accZ[i] = az * scaleI;
                const double ratio = 1.0 - pressureDepletion;
                vacuumPressureRatio[i] = (ratio > 0.0) ? ratio : 0.0;
            }
        }

        __global__ void SynthesisKickDriftKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            const double* __restrict__ vacuumPressureRatio,
            const double* __restrict__ vibrationalDensity,
            double* __restrict__ localProperTime,
            std::size_t bodyCount,
            double dt,
            double harameinOmegaZ,
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
            double vx = velX[i] + (halfDt * accX[i]);
            double vy = velY[i] + (halfDt * accY[i]);
            const double vz = velZ[i] + (halfDt * accZ[i]);

            // Symplectic Strang-split Haramein-Rauscher spacetime-torque Coriolis half-rotation.
            if (harameinOmegaZ != 0.0) {
                const double theta = harameinOmegaZ * dt;
                const double c = cos(theta);
                const double s = sin(theta);
                const double rvx = (vx * c) + (vy * s);
                const double rvy = (-vx * s) + (vy * c);
                vx = rvx;
                vy = rvy;
            }

            velX[i] = vx; velY[i] = vy; velZ[i] = vz;
            posX[i] += dt * vx; posY[i] += dt * vy; posZ[i] += dt * vz;

            const double rate = nonLinearTimeEnabled
                ? ComputeLocalTimeRate(vacuumPressureRatio[i], vibrationalDensity[i], baseDensity)
                : 1.0;
            localProperTime[i] += dt * rate;
        }

        __global__ void SynthesisFinalKickKernel(
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            std::size_t bodyCount,
            double dt,
            double harameinOmegaZ
        ) {
            const std::size_t i =
                static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                static_cast<std::size_t>(threadIdx.x);
            if (i >= bodyCount) {
                return;
            }

            double vx = velX[i];
            double vy = velY[i];
            if (harameinOmegaZ != 0.0) {
                const double theta = harameinOmegaZ * dt;
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

        __global__ void SynthesisEulerStepKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
            double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
            const double* __restrict__ accX, const double* __restrict__ accY, const double* __restrict__ accZ,
            const double* __restrict__ vacuumPressureRatio,
            const double* __restrict__ vibrationalDensity,
            double* __restrict__ localProperTime,
            std::size_t bodyCount,
            double dt,
            double harameinOmegaZ,
            bool nonLinearTimeEnabled,
            double baseDensity
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
            posX[i] += dt * vx; posY[i] += dt * vy; posZ[i] += dt * vz;
            velX[i] = vx + dt * (accX[i] + ComputeHarameinCoriolisAccelX(vy, harameinOmegaZ));
            velY[i] = vy + dt * (accY[i] + ComputeHarameinCoriolisAccelY(vx, harameinOmegaZ));
            velZ[i] = vz + dt * accZ[i];
            const double rate = nonLinearTimeEnabled
                ? ComputeLocalTimeRate(vacuumPressureRatio[i], vibrationalDensity[i], baseDensity)
                : 1.0;
            localProperTime[i] += dt * rate;
        }

    } // anonymous namespace

    UnifiedSynthesisSolver::~UnifiedSynthesisSolver() {
        Shutdown();
    }

    bool UnifiedSynthesisSolver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        Shutdown();
        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err) || scene.bodies.empty()) {
            spdlog::error("UnifiedSynthesisSolver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_bodyCount = scene.bodies.size();
        m_integrator = scene.physicsProfile.integrator;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_unifiedParams = scene.physicsProfile.unifiedField;

        // Evaluate Pais Eq. (1) HEEMFG flux & Vogel/Willis 4096 Hz drive power from shared Thinker headers.
        const auto& p = m_unifiedParams.pais;
        m_paisMaxPoyntingFlux = ComputePaisMaxPoyntingFlux(
            p.geometricFactor, p.surfaceChargeDensity, p.rotationRadius,
            p.angularFrequencyRadS, p.vibrationAmplitude, p.vibrationFrequencyHz, p.translationSpeed
        );
        m_paisInertialMassFactor = ComputePaisInertialMassFactor(
            m_paisMaxPoyntingFlux, p.inertialReductionCoupling
        );
        m_paisAccelAmplification = ComputePaisAccelAmplification(
            m_paisMaxPoyntingFlux, p.inertialReductionCoupling
        );

        const auto& vw = m_unifiedParams.vogelWillis;
        const double geomRes = ComputeVogelGeometricResonanceFactor(
            vw.baseTerminationAngleDeg, vw.topTerminationAngleDeg
        );
        m_vogelDrivePower = ComputeVogelTorsionDrivePower(
            vw.torsionCoilFrequencyHz, vw.driveVoltageV, geomRes
        );

        m_totalBodyPhryll = 0.0;
        m_cumulativeCrystalEnergy = 0.0;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        if (!m_posX.Allocate(context, m_bodyCount) || !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) || !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) || !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) || !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) || !m_mass.Allocate(context, m_bodyCount) ||
            !m_radius.Allocate(context, m_bodyCount) ||
            !m_vibrationalDensity.Allocate(context, m_bodyCount) ||
            !m_coherence.Allocate(context, m_bodyCount) ||
            !m_phryllCharge.Allocate(context, m_bodyCount) ||
            !m_accelScale.Allocate(context, m_bodyCount) ||
            !m_vacuumPressureRatio.Allocate(context, m_bodyCount) ||
            !m_localProperTime.Allocate(context, m_bodyCount)) {
            Shutdown();
            return false;
        }

        m_hostPosX.resize(m_bodyCount); m_hostPosY.resize(m_bodyCount); m_hostPosZ.resize(m_bodyCount);
        m_hostVelX.resize(m_bodyCount); m_hostVelY.resize(m_bodyCount); m_hostVelZ.resize(m_bodyCount);
        m_hostMass.resize(m_bodyCount); m_hostRadius.resize(m_bodyCount);
        m_hostDensity.resize(m_bodyCount); m_hostCoherence.resize(m_bodyCount);
        m_hostPhryll.resize(m_bodyCount);
        m_hostAccelScale.assign(m_bodyCount, 1.0);
        m_hostPressureRatio.assign(m_bodyCount, 1.0);
        m_hostProperTime.assign(m_bodyCount, 0.0);

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const auto& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x; m_hostPosY[i] = b.position.y; m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x; m_hostVelY[i] = b.velocity.y; m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;
            m_hostRadius[i] = b.radius;
            m_hostDensity[i] = b.vibrationalDensity;
            m_hostCoherence[i] = b.consciousnessCoherence;
            m_hostPhryll[i] = b.phryllCharge;
            m_hostAccelScale[i] = (b.category == "HEEMFG_Craft") ? m_paisAccelAmplification : 1.0;
            m_totalBodyPhryll += std::abs(b.phryllCharge);
        }

        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) || !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) || !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) || !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) || !m_radius.CopyFromHost(m_hostRadius, m_stream) ||
            !m_vibrationalDensity.CopyFromHost(m_hostDensity, m_stream) ||
            !m_coherence.CopyFromHost(m_hostCoherence, m_stream) ||
            !m_phryllCharge.CopyFromHost(m_hostPhryll, m_stream) ||
            !m_accelScale.CopyFromHost(m_hostAccelScale, m_stream) ||
            !m_accX.ZeroFill(m_stream) || !m_accY.ZeroFill(m_stream) || !m_accZ.ZeroFill(m_stream) ||
            !m_vacuumPressureRatio.CopyFromHost(m_hostPressureRatio, m_stream) ||
            !m_localProperTime.ZeroFill(m_stream)) {
            Shutdown();
            return false;
        }

        if (!LaunchSynthesisFieldKernel() || !Synchronize()) {
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

        spdlog::info(
            "UnifiedSynthesisSolver initialized for '{}' ({} bodies, EssonneVortex={}, HarameinOmegaZ={:.3g}, PaisFactor={:.4g}, Love={:.3g}, VogelPhryll={:.3g}).",
            scene.name, m_bodyCount,
            m_unifiedParams.aetherVortexEnabled ? "ON" : "OFF",
            m_unifiedParams.haramein.spacetimeTorqueOmegaZ,
            m_paisInertialMassFactor,
            m_unifiedParams.loveBindingCoupling,
            m_unifiedParams.vogelWillis.phryllCoupling
        );
        return true;
    }

    void UnifiedSynthesisSolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_posX.Release(); m_posY.Release(); m_posZ.Release();
        m_velX.Release(); m_velY.Release(); m_velZ.Release();
        m_accX.Release(); m_accY.Release(); m_accZ.Release();
        m_mass.Release(); m_radius.Release();
        m_vibrationalDensity.Release(); m_coherence.Release(); m_phryllCharge.Release();
        m_accelScale.Release(); m_vacuumPressureRatio.Release(); m_localProperTime.Release();
        m_context = nullptr; m_stream = nullptr; m_initialized = false;
        m_bodyCount = 0U; m_stepCount = 0ULL; m_simulationTime = 0.0;
        m_initialTotalEnergy = 0.0; m_cumulativeCrystalEnergy = 0.0;
    }

    bool UnifiedSynthesisSolver::LaunchSynthesisFieldKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }
        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kFieldTileSize);
        if (gridSize == 0U) {
            return true;
        }
        ComputeUnifiedSynthesisKernel<kFieldTileSize><<<gridSize, kFieldTileSize, 0, m_stream>>>(
            m_posX.Data(), m_posY.Data(), m_posZ.Data(), m_mass.Data(),
            m_coherence.Data(), m_phryllCharge.Data(), m_accelScale.Data(),
            m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_vacuumPressureRatio.Data(),
            m_bodyCount, m_gravitationalConstant, m_softeningLength * m_softeningLength,
            m_unifiedParams.referenceLightSpeed,
            m_unifiedParams.aetherVortexEnabled,
            m_unifiedParams.aetherVortexCriticalRadius,
            m_unifiedParams.aetherVortexScaleHeight,
            m_unifiedParams.aetherVortexVerticalFieldGz,
            m_unifiedParams.loveBindingCoupling,
            m_unifiedParams.vogelWillis.phryllCoupling,
            m_vogelDrivePower
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool UnifiedSynthesisSolver::Step(double dt) {
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
            SynthesisKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_vacuumPressureRatio.Data(), m_vibrationalDensity.Data(), m_localProperTime.Data(),
                m_bodyCount, dt, m_unifiedParams.haramein.spacetimeTorqueOmegaZ,
                m_unifiedParams.nonLinearTimeEnabled, m_unifiedParams.baseVibrationalDensity
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchSynthesisFieldKernel()) {
                return false;
            }
            SynthesisFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_bodyCount, dt, m_unifiedParams.haramein.spacetimeTorqueOmegaZ
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            SynthesisEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(),
                m_vacuumPressureRatio.Data(), m_vibrationalDensity.Data(), m_localProperTime.Data(),
                m_bodyCount, dt, m_unifiedParams.haramein.spacetimeTorqueOmegaZ,
                m_unifiedParams.nonLinearTimeEnabled, m_unifiedParams.baseVibrationalDensity
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchSynthesisFieldKernel()) {
                return false;
            }
        }

        if (m_unifiedParams.vogelWillis.phryllCoupling != 0.0 || m_unifiedParams.phryllFieldAmplitude > 0.0) {
            m_cumulativeCrystalEnergy += m_vogelDrivePower * std::max(1.0, m_totalBodyPhryll) * std::abs(dt);
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool UnifiedSynthesisSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool UnifiedSynthesisSolver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool UnifiedSynthesisSolver::DownloadStateToStaging() {
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

    bool UnifiedSynthesisSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool UnifiedSynthesisSolver::GetLocalProperTimes(std::vector<double>& outProperTimes) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outProperTimes = m_hostProperTime;
        return true;
    }

    bool UnifiedSynthesisSolver::GetVacuumPressureRatios(std::vector<double>& outPressureRatios) {
        if (!DownloadStateToStaging()) {
            return false;
        }
        outPressureRatios = m_hostPressureRatio;
        return true;
    }

    bool UnifiedSynthesisSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0, potential = 0.0, totalMass = 0.0;
        scene::Vec3d linearMomentum{}, angularMomentum{}, weightedPos{};
        double sumPressure = 0.0, sumLight = 0.0, sumRate = 0.0, sumProper = 0.0, sumPhi = 0.0;
        double totalPhryll = m_unifiedParams.phryllFieldAmplitude;
        const double softeningSq = m_softeningLength * m_softeningLength;

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const double mi = m_hostMass[i];
            const double mEff = mi / m_hostAccelScale[i];
            const scene::Vec3d ri{m_hostPosX[i], m_hostPosY[i], m_hostPosZ[i]};
            const scene::Vec3d vi{m_hostVelX[i], m_hostVelY[i], m_hostVelZ[i]};

            totalMass += mi;
            kinetic += 0.5 * mEff * vi.NormSquared();
            const scene::Vec3d pi = vi * mEff;
            linearMomentum = linearMomentum + pi;
            angularMomentum = angularMomentum + scene::Vec3d::Cross(ri, pi);
            weightedPos = weightedPos + (ri * mi);

            const double pRatio = m_hostPressureRatio[i];
            sumPressure += pRatio;
            sumLight += (pRatio > 0.0) ? std::sqrt(pRatio) : 0.0;
            sumRate += m_unifiedParams.nonLinearTimeEnabled
                ? ComputeLocalTimeRate(pRatio, m_hostDensity[i], m_unifiedParams.baseVibrationalDensity)
                : 1.0;
            sumProper += m_hostProperTime[i];
            sumPhi += ComputeHolographicRatioPhi(m_hostRadius[i]);
            totalPhryll += m_hostPhryll[i];

            if (m_unifiedParams.aetherVortexEnabled && i > 0U) {
                const double dx = ri.x - m_hostPosX[0];
                const double dy = ri.y - m_hostPosY[0];
                potential += mi * ComputeEssonneAetherVortexPotential(
                    std::sqrt((dx * dx) + (dy * dy)),
                    m_hostMass[0], m_gravitationalConstant,
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
                    if (m_unifiedParams.vogelWillis.phryllCoupling != 0.0 && m_vogelDrivePower > 0.0) {
                        potential -= (m_unifiedParams.vogelWillis.phryllCoupling * m_vogelDrivePower *
                                      m_hostPhryll[i] * m_hostPhryll[j]) / dist;
                    }
                }
            }
        }

        const double totalE = kinetic + potential;
        const double baselineE = (m_stepCount == 0ULL && m_initialTotalEnergy == 0.0) ? totalE : m_initialTotalEnergy;
        const double invN = (m_bodyCount > 0U) ? (1.0 / static_cast<double>(m_bodyCount)) : 0.0;
        const double satRatio = ComputeVogelCrystalSaturationRatio(
            m_cumulativeCrystalEnergy, m_unifiedParams.vogelWillis.saturationThreshold
        );

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::UnifiedSynthesis;
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
        outDiagnostics.meanLocalTimeRate = sumRate * invN;
        outDiagnostics.meanLocalProperTime = sumProper * invN;
        outDiagnostics.meanHolographicPhi = sumPhi * invN;
        outDiagnostics.harameinSpacetimeTorqueOmegaZ = m_unifiedParams.haramein.spacetimeTorqueOmegaZ;
        outDiagnostics.paisMaxPoyntingFlux = m_paisMaxPoyntingFlux;
        outDiagnostics.paisInertialMassFactor = m_paisInertialMassFactor;
        outDiagnostics.totalPhryllFlux = totalPhryll;
        outDiagnostics.vogelCrystalSaturationRatio = satRatio;
        outDiagnostics.vogelCrystalSinging = (satRatio >= 1.0);
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::unified
