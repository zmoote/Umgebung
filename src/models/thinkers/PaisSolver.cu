/**
 * @file PaisSolver.cu
 * @brief FP64 CUDA implementation of the standalone `Thinker::Pais` module
 *        (`US10144532.pdf` HEEMFG Eq. 1 Poynting energy flux & vacuum-polarized
 *        inertial mass reduction $m_{\text{eff}} = m_0 / (1 + \kappa S_{\max})$).
 */

#include "umgebung/models/thinkers/PaisSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/thinkers/PaisModel.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::thinkers {

    namespace {

        template <int kTileSize>
        __global__ void ComputePaisAccelKernel(
            const double* __restrict__ posX,
            const double* __restrict__ posY,
            const double* __restrict__ posZ,
            const double* __restrict__ mass,
            const double* __restrict__ accelScale,
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

            double xi = 0.0, yi = 0.0, zi = 0.0, scaleI = 1.0;
            if (i < bodyCount) {
                xi = posX[i];
                yi = posY[i];
                zi = posZ[i];
                scaleI = accelScale[i];
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

            // Apply Pais HEEMFG inertial mass reduction: a_eff = F / m_eff = (m_0 / m_eff) * a_0.
            if (i < bodyCount) {
                accX[i] = ax * scaleI;
                accY[i] = ay * scaleI;
                accZ[i] = az * scaleI;
            }
        }

        __global__ void PaisKickDriftKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
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
            const double vxHalf = velX[i] + (halfDt * accX[i]);
            const double vyHalf = velY[i] + (halfDt * accY[i]);
            const double vzHalf = velZ[i] + (halfDt * accZ[i]);
            velX[i] = vxHalf; velY[i] = vyHalf; velZ[i] = vzHalf;
            posX[i] += dt * vxHalf; posY[i] += dt * vyHalf; posZ[i] += dt * vzHalf;
        }

        __global__ void PaisFinalKickKernel(
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

        __global__ void PaisEulerStepKernel(
            double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
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
            posX[i] += dt * velX[i]; posY[i] += dt * velY[i]; posZ[i] += dt * velZ[i];
            velX[i] += dt * accX[i]; velY[i] += dt * accY[i]; velZ[i] += dt * accZ[i];
        }

    } // anonymous namespace

    PaisSolver::~PaisSolver() {
        Shutdown();
    }

    bool PaisSolver::Initialize(compute::CUDAContext& context, const scene::UniverseScene& scene) {
        Shutdown();
        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err) || scene.bodies.empty()) {
            spdlog::error("PaisSolver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_bodyCount = scene.bodies.size();
        m_integrator = scene.physicsProfile.integrator;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_softeningLength = scene.physicsProfile.mainstreamGravity.softeningLength;
        m_paisParams = scene.physicsProfile.unifiedField.pais;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;

        m_maxPoyntingFlux = ComputePaisMaxPoyntingFlux(
            m_paisParams.geometricFactor,
            m_paisParams.surfaceChargeDensity,
            m_paisParams.rotationRadius,
            m_paisParams.angularFrequencyRadS,
            m_paisParams.vibrationAmplitude,
            m_paisParams.vibrationFrequencyHz,
            m_paisParams.translationSpeed
        );
        m_inertialMassFactor = ComputePaisInertialMassFactor(
            m_maxPoyntingFlux, m_paisParams.inertialReductionCoupling
        );
        m_accelAmplification = ComputePaisAccelAmplification(
            m_maxPoyntingFlux, m_paisParams.inertialReductionCoupling
        );

        if (!m_posX.Allocate(context, m_bodyCount) || !m_posY.Allocate(context, m_bodyCount) ||
            !m_posZ.Allocate(context, m_bodyCount) || !m_velX.Allocate(context, m_bodyCount) ||
            !m_velY.Allocate(context, m_bodyCount) || !m_velZ.Allocate(context, m_bodyCount) ||
            !m_accX.Allocate(context, m_bodyCount) || !m_accY.Allocate(context, m_bodyCount) ||
            !m_accZ.Allocate(context, m_bodyCount) || !m_mass.Allocate(context, m_bodyCount) ||
            !m_accelScale.Allocate(context, m_bodyCount)) {
            Shutdown();
            return false;
        }

        bool hasExplicitHeemfgCraft = false;
        for (const auto& b : scene.bodies) {
            if (b.category == "HEEMFG_Craft") {
                hasExplicitHeemfgCraft = true;
                break;
            }
        }

        m_hostPosX.resize(m_bodyCount);
        m_hostPosY.resize(m_bodyCount);
        m_hostPosZ.resize(m_bodyCount);
        m_hostVelX.resize(m_bodyCount);
        m_hostVelY.resize(m_bodyCount);
        m_hostVelZ.resize(m_bodyCount);
        m_hostMass.resize(m_bodyCount);
        m_hostAccelScale.assign(m_bodyCount, 1.0);

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            const auto& b = scene.bodies[i];
            m_hostPosX[i] = b.position.x;
            m_hostPosY[i] = b.position.y;
            m_hostPosZ[i] = b.position.z;
            m_hostVelX[i] = b.velocity.x;
            m_hostVelY[i] = b.velocity.y;
            m_hostVelZ[i] = b.velocity.z;
            m_hostMass[i] = b.mass;

            const bool isHeemfgBody = hasExplicitHeemfgCraft
                ? (b.category == "HEEMFG_Craft")
                : (i > 0U);
            m_hostAccelScale[i] = isHeemfgBody ? m_accelAmplification : 1.0;
        }

        if (!m_posX.CopyFromHost(m_hostPosX, m_stream) || !m_posY.CopyFromHost(m_hostPosY, m_stream) ||
            !m_posZ.CopyFromHost(m_hostPosZ, m_stream) || !m_velX.CopyFromHost(m_hostVelX, m_stream) ||
            !m_velY.CopyFromHost(m_hostVelY, m_stream) || !m_velZ.CopyFromHost(m_hostVelZ, m_stream) ||
            !m_mass.CopyFromHost(m_hostMass, m_stream) || !m_accelScale.CopyFromHost(m_hostAccelScale, m_stream) ||
            !m_accX.ZeroFill(m_stream) || !m_accY.ZeroFill(m_stream) || !m_accZ.ZeroFill(m_stream)) {
            Shutdown();
            return false;
        }

        if (!LaunchPaisAccelKernel() || !Synchronize()) {
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

    void PaisSolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_posX.Release(); m_posY.Release(); m_posZ.Release();
        m_velX.Release(); m_velY.Release(); m_velZ.Release();
        m_accX.Release(); m_accY.Release(); m_accZ.Release();
        m_mass.Release(); m_accelScale.Release();
        m_context = nullptr; m_stream = nullptr; m_initialized = false;
        m_bodyCount = 0U; m_stepCount = 0ULL; m_simulationTime = 0.0; m_initialTotalEnergy = 0.0;
    }

    bool PaisSolver::LaunchPaisAccelKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }
        const unsigned int gridSize = compute::ComputeGridSize1D(m_bodyCount, kTileSize);
        if (gridSize == 0U) {
            return true;
        }
        ComputePaisAccelKernel<kTileSize><<<gridSize, kTileSize, 0, m_stream>>>(
            m_posX.Data(), m_posY.Data(), m_posZ.Data(), m_mass.Data(), m_accelScale.Data(),
            m_accX.Data(), m_accY.Data(), m_accZ.Data(),
            m_bodyCount, m_gravitationalConstant, m_softeningLength * m_softeningLength
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool PaisSolver::Step(double dt) {
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
            PaisKickDriftKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchPaisAccelKernel()) {
                return false;
            }
            PaisFinalKickKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
                return false;
            }
        } else {
            PaisEulerStepKernel<<<gridSize, blockSize, 0, m_stream>>>(
                m_posX.Data(), m_posY.Data(), m_posZ.Data(),
                m_velX.Data(), m_velY.Data(), m_velZ.Data(),
                m_accX.Data(), m_accY.Data(), m_accZ.Data(), m_bodyCount, dt
            );
            if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR() || !LaunchPaisAccelKernel()) {
                return false;
            }
        }

        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool PaisSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool PaisSolver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool PaisSolver::DownloadStateToStaging() {
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

    bool PaisSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
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

    bool PaisSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!DownloadStateToStaging()) {
            return false;
        }

        double kinetic = 0.0, potential = 0.0, totalMass = 0.0;
        scene::Vec3d linearMomentum{}, angularMomentum{}, weightedPos{};
        const double softeningSq = m_softeningLength * m_softeningLength;

        for (std::size_t i = 0U; i < m_bodyCount; ++i) {
            // Effective inertial mass m_eff = m_0 / accelScale_i captures conserved
            // Hamiltonian energy T_eff = sum 0.5 * m_eff * v^2 + U_grav for Pais craft!
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

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::ThinkerPais;
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
        outDiagnostics.paisMaxPoyntingFlux = m_maxPoyntingFlux;
        outDiagnostics.paisInertialMassFactor = m_inertialMassFactor;
        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::thinkers
