/**
 * @file FieldGrid2DSolver.cu
 * @brief FP64 CUDA 2D spatial field & TMz Yee-lattice FDTD implementation of
 *        `umgebung::models::FieldGrid2DSolver` for `SimulationDomain::FieldGrid2D`.
 *
 * @details
 * Implements GPU kernels for:
 *   1. **2D TMz Yee-Lattice FDTD Electrodynamics (`Mainstream::MaxwellEM`):**
 *      Evolves $(E_z, H_x, H_y)$ in IEEE-754 double precision with strict CFL
 *      stability enforcement ($S_{\text{CFL}} = c\,\Delta t_{\text{eff}}/\Delta x \le 0.45 < 1/\sqrt{2}$,
 *      formally proven in `formal/Umgebung/Numerical/FdtdStability.lean`) and a
 *      smooth absorbing boundary sponge layer.
 *   2. **Salvatore Pais HEEMFG Resonant Cavity (`Thinker::Pais`, `US10144532.pdf`):**
 *      Drives the stationary spinning/vibrating charged shell ring cavity with
 *      Pais's vacuum-polarization amplification factor $(1 + \kappa S_{\max})$
 *      derived verbatim from `ComputePaisMaxPoyntingFlux` (Eq. 1).
 *   3. **Vogel / Willis / Eredyon 4096 Hz Phryll Crystal Cavity (`Thinker::VogelWillis`):**
 *      Models the stationary double-terminated $51.84^\circ / 60.0^\circ$ quartz
 *      crystal dielectric core driven by orthogonal copper/silver torsion coils
 *      pulsing $180^\circ$ out of phase (`PhryllGenInstructions.pdf`), tracking
 *      live Phryll field intensity, probe waveform, crystal saturation ratio, and
 *      the audible "singing" threshold warning.
 *   4. **Chris Essonne 2D Spatial Vacuum Pressure Field (`Thinker::Essonne`, `8052.pdf`):**
 *      Evaluates $P(x,y)/P_0$ (interior Eq. 35 + exterior Eq. 21) and local
 *      light speed $c'(x,y)/c$ (Eq. 27) across every cell of the 2D GPU grid.
 *   5. **Unified Synthesis 2D Cavity (`Unified::Synthesis`):**
 *      Composes vacuum refractive-index modulation $c'(x,y)/c$, Pais vacuum
 *      polarization, and Vogel/Willis crystal/coil resonance in one GPU lattice.
 */

#include "umgebung/models/FieldGrid2DSolver.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/models/thinkers/EssonneModel.hpp"
#include "umgebung/models/thinkers/PaisModel.hpp"
#include "umgebung/models/thinkers/VogelWillisModel.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models {

    namespace {

        /** @brief Smooth quadratic absorbing sponge attenuation near the outer 10-cell border. */
        __device__ inline double ComputeSpongeFactor(
            std::uint32_t ix,
            std::uint32_t iy,
            std::uint32_t width,
            std::uint32_t height,
            double baseDamping
        ) noexcept {
            constexpr std::uint32_t kSpongeWidth = 10U;
            const std::uint32_t distX = (ix < (width - 1U - ix)) ? ix : (width - 1U - ix);
            const std::uint32_t distY = (iy < (height - 1U - iy)) ? iy : (height - 1U - iy);
            const std::uint32_t minDist = (distX < distY) ? distX : distY;
            double damp = baseDamping;
            if (minDist < kSpongeWidth) {
                const double s = static_cast<double>(kSpongeWidth - minDist) / static_cast<double>(kSpongeWidth);
                damp += 0.12 * s * s;
            }
            const double factor = 1.0 - damp;
            return (factor > 0.0) ? factor : 0.0;
        }

        /**
         * @brief Initializes the 2D inverse relative permittivity map `invEps(x,y) = 1 / eps_r(x,y)`
         *        encoding stationary quartz crystal facets (`51.84° / 60.0°`) or cavity walls.
         */
        __global__ void InitCavityPermittivityKernel(
            double* __restrict__ invEps,
            std::uint32_t width,
            std::uint32_t height,
            double halfExtent,
            bool hasVogelCrystal,
            double crystalRadius,
            double baseAngleRad,
            double topAngleRad,
            bool hasEssonneRefraction,
            double essonneRs
        ) {
            const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= width || iy >= height) {
                return;
            }

            const double dx = (2.0 * halfExtent) / static_cast<double>(width - 1U);
            const double dy = (2.0 * halfExtent) / static_cast<double>(height - 1U);
            const double x = -halfExtent + static_cast<double>(ix) * dx;
            const double y = -halfExtent + static_cast<double>(iy) * dy;

            double epsR = 1.0;

            // Stationary double-terminated Vogel/Eredyon quartz crystal silhouette:
            // Lower half tapers at 51.84 deg, upper half tapers at 60.0 deg (`PhryllGenInstructions.pdf`).
            if (hasVogelCrystal && crystalRadius > 0.0) {
                const double halfWidth = crystalRadius * 0.65;
                const double slope = (y >= 0.0) ? tan(topAngleRad) : tan(baseAngleRad);
                const double maxY = halfWidth * slope;
                const double absX = fabs(x);
                const double absY = fabs(y);
                if (maxY > 0.0 && absY <= maxY && absX <= halfWidth * (1.0 - (absY / maxY))) {
                    // Fused quartz relative permittivity eps_r ~ 3.8..4.5.
                    epsR = 4.2;
                }
            }

            // Optional Essonne vacuum refractive index modulation n(r)^2 = 1 / (1 - R_s / r) (`8052.pdf` Eq. 23).
            if (hasEssonneRefraction && essonneRs > 0.0) {
                const double r = sqrt((x * x) + (y * y));
                const double rSafe = (r > 1.05 * essonneRs) ? r : (1.05 * essonneRs);
                const double pRatio = thinkers::ComputeEssonneVacuumPressureRatio(rSafe, essonneRs);
                if (pRatio > 0.05) {
                    epsR *= (1.0 / pRatio);
                }
            }

            invEps[iy * width + ix] = 1.0 / epsR;
        }

        /**
         * @brief 2D TMz Yee-lattice FDTD Stage 1: Magnetic field `(Hx, Hy)` curl update.
         */
        __global__ void FdtdUpdateHKernel(
            const double* __restrict__ ez,
            double* __restrict__ hx,
            double* __restrict__ hy,
            std::uint32_t width,
            std::uint32_t height,
            double cflFactor,
            double baseDamping
        ) {
            const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= width || iy >= height) {
                return;
            }

            const std::size_t idx = static_cast<std::size_t>(iy) * width + ix;
            const double damp = ComputeSpongeFactor(ix, iy, width, height, baseDamping);

            if (iy + 1U < height) {
                const double dEzDy = ez[(iy + 1U) * width + ix] - ez[idx];
                hx[idx] = (damp * hx[idx]) - (cflFactor * dEzDy);
            } else {
                hx[idx] = 0.0;
            }

            if (ix + 1U < width) {
                const double dEzDx = ez[iy * width + (ix + 1U)] - ez[idx];
                hy[idx] = (damp * hy[idx]) + (cflFactor * dEzDx);
            } else {
                hy[idx] = 0.0;
            }
        }

        /**
         * @brief 2D TMz Yee-lattice FDTD Stage 2: Electric field `Ez` curl update,
         *        stationary device source injection (Pais rotating/vibrating cavity
         *        ring OR Vogel/Willis 180-deg orthogonal 4096 Hz torsion coils), and
         *        local field energy density / Poynting intensity output.
         */
        __global__ void FdtdUpdateEzAndVisualKernel(
            double* __restrict__ ez,
            const double* __restrict__ hx,
            const double* __restrict__ hy,
            const double* __restrict__ invEps,
            double* __restrict__ visualScalar,
            std::uint32_t width,
            std::uint32_t height,
            double halfExtent,
            double cflFactor,
            double dtEff,
            double baseDamping,
            double simTime,
            double sourceFreq,
            int modeKind,
            double sourceAmplitude,
            double cavityRadius,
            double coilOffset
        ) {
            const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= width || iy >= height) {
                return;
            }

            const std::size_t idx = static_cast<std::size_t>(iy) * width + ix;
            const double dx = (2.0 * halfExtent) / static_cast<double>(width - 1U);
            const double dy = (2.0 * halfExtent) / static_cast<double>(height - 1U);
            const double x = -halfExtent + static_cast<double>(ix) * dx;
            const double y = -halfExtent + static_cast<double>(iy) * dy;

            double newEz = 0.0;
            if (ix > 0U && iy > 0U && ix + 1U < width && iy + 1U < height) {
                const double damp = ComputeSpongeFactor(ix, iy, width, height, baseDamping);
                const double curlH =
                    (hy[idx] - hy[iy * width + (ix - 1U)]) -
                    (hx[idx] - hx[(iy - 1U) * width + ix]);
                newEz = (damp * ez[idx]) + (invEps[idx] * cflFactor * curlH);

                const double omega = 2.0 * thinkers::constants::kPi * sourceFreq;
                double srcJ = 0.0;

                if (modeKind == 1) {
                    // Pais HEEMFG Cavity (or MaxwellEM comparison on Pais cavity):
                    // Stationary spinning & harmonically vibrating charged shell ring at r = cavityRadius.
                    const double r = sqrt((x * x) + (y * y));
                    const double phi = atan2(y, x);
                    const double ringSigma = 0.14 * halfExtent;
                    const double dr = r - cavityRadius;
                    const double ringEnv = exp(-(dr * dr) / (2.0 * ringSigma * ringSigma));
                    // Traveling azimuthal spin + harmonic radial breathing mode (US10144532.pdf).
                    const double wave = sin((omega * simTime) - (2.0 * phi)) + 0.5 * cos(2.0 * omega * simTime);
                    srcJ = sourceAmplitude * ringEnv * wave;
                } else if (modeKind == 2) {
                    // Vogel / Willis / Eredyon 4096 Hz Orthogonal Copper/Silver Torsion Coils:
                    // Coil pair A along X (+/- coilOffset, 0) at phase 0 deg;
                    // Coil pair B along Y (0, +/- coilOffset) at alternating 180 deg phase shift!
                    const double coilSigma = 0.11 * halfExtent;
                    const double inv2SigSq = 1.0 / (2.0 * coilSigma * coilSigma);
                    const double gX1 = exp(-(((x - coilOffset) * (x - coilOffset)) + (y * y)) * inv2SigSq);
                    const double gX2 = exp(-(((x + coilOffset) * (x + coilOffset)) + (y * y)) * inv2SigSq);
                    const double gY1 = exp(-((x * x) + ((y - coilOffset) * (y - coilOffset))) * inv2SigSq);
                    const double gY2 = exp(-((x * x) + ((y + coilOffset) * (y + coilOffset))) * inv2SigSq);

                    const double pulseA = sin(omega * simTime);
                    const double pulseB = sin((omega * simTime) + thinkers::constants::kPi); // Exact 180 deg phase shift
                    srcJ = sourceAmplitude * (((gX1 + gX2) * pulseA) + ((gY1 + gY2) * pulseB));
                } else {
                    // Generic centered harmonic EM dipole cavity baseline.
                    const double sigma = 0.12 * halfExtent;
                    const double env = exp(-((x * x) + (y * y)) / (2.0 * sigma * sigma));
                    srcJ = sourceAmplitude * env * sin(omega * simTime);
                }

                newEz += dtEff * srcJ;
            }

            ez[idx] = newEz;

            // Compute smooth non-negative instantaneous local field intensity / energy density:
            // u(x,y) = 0.5 * (Ez^2 + Hx^2 + Hy^2) + |Ez| * sqrt(Hx^2 + Hy^2) (Poynting flux magnitude).
            const double hMagSq = (hx[idx] * hx[idx]) + (hy[idx] * hy[idx]);
            const double poyntingMag = fabs(newEz) * sqrt(hMagSq);
            const double energyDens = 0.5 * ((newEz * newEz) + hMagSq);
            visualScalar[idx] = energyDens + poyntingMag;
        }

        /**
         * @brief Dedicated CUDA kernel for `Thinker::Essonne` 2D spatial vacuum
         *        pressure ratio `P(x,y)/P_0` (`8052.pdf` Eq. 21 exterior & Eq. 35 interior).
         */
        __global__ void ComputeEssonneVacuumGrid2DKernel(
            double* __restrict__ visualScalar,
            double* __restrict__ ezOut,
            std::uint32_t width,
            std::uint32_t height,
            double halfExtent,
            double bodyRadius,
            double schwarzschildRadius
        ) {
            const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= width || iy >= height) {
                return;
            }
            const double dx = (2.0 * halfExtent) / static_cast<double>(width - 1U);
            const double dy = (2.0 * halfExtent) / static_cast<double>(height - 1U);
            const double x = -halfExtent + static_cast<double>(ix) * dx;
            const double y = -halfExtent + static_cast<double>(iy) * dy;
            const double r = sqrt((x * x) + (y * y));

            double pRatio = 1.0;
            if (r <= bodyRadius && bodyRadius > 0.0) {
                pRatio = thinkers::ComputeEssonneInteriorPressureRatio(r, bodyRadius, schwarzschildRadius);
            } else if (r > 0.0) {
                pRatio = thinkers::ComputeEssonneVacuumPressureRatio(r, schwarzschildRadius);
            }
            if (pRatio < 0.0) {
                pRatio = 0.0;
            }

            const std::size_t idx = static_cast<std::size_t>(iy) * width + ix;
            visualScalar[idx] = pRatio;
            ezOut[idx] = sqrt(pRatio); // Local light speed ratio c'(x,y)/c (`8052.pdf` Eq. 27).
        }

    } // anonymous namespace

    FieldGrid2DSolver::FieldGrid2DSolver(scene::ModelId modelId)
        : m_modelId(modelId) {
        switch (m_modelId) {
            case scene::ModelId::MainstreamMaxwellEM:
                m_profileType = scene::PhysicsProfileType::Mainstream;
                m_solverName = "Mainstream::MaxwellEM (2D TMz Yee FDTD, FP64 CUDA)";
                break;
            case scene::ModelId::ThinkerPais:
                m_profileType = scene::PhysicsProfileType::Custom;
                m_solverName = "Thinker::Pais (HEEMFG Resonant Cavity 2D GPU Field)";
                break;
            case scene::ModelId::ThinkerVogelWillis:
                m_profileType = scene::PhysicsProfileType::Custom;
                m_solverName = "Thinker::VogelWillis (4096 Hz Phryll Crystal 2D GPU Field)";
                break;
            case scene::ModelId::ThinkerEssonne:
                m_profileType = scene::PhysicsProfileType::Custom;
                m_solverName = "Thinker::Essonne (2D Spatial Vacuum Pressure P/P0 Field)";
                break;
            case scene::ModelId::UnifiedSynthesis:
                m_profileType = scene::PhysicsProfileType::Unified;
                m_solverName = "Unified::Synthesis (Composed 2D Vacuum + Pais + Vogel Cavity)";
                break;
            default:
                m_profileType = scene::PhysicsProfileType::Mainstream;
                m_solverName = "FieldGrid2D FP64 CUDA Solver";
                break;
        }
    }

    FieldGrid2DSolver::~FieldGrid2DSolver() {
        Shutdown();
    }

    bool FieldGrid2DSolver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        Shutdown();

        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err)) {
            spdlog::error("FieldGrid2DSolver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_gridConfig = scene.physicsProfile.fieldGrid2D;
        m_unifiedParams = scene.physicsProfile.unifiedField;
        m_gravitationalConstant = scene.physicsProfile.mainstreamGravity.gravitationalConstant;
        m_integrator = scene.physicsProfile.integrator;
        m_stationaryAnchors = scene.bodies;

        m_width = std::max(16U, m_gridConfig.gridWidth);
        m_height = std::max(16U, m_gridConfig.gridHeight);
        m_cellCount = static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height);
        m_dx = (2.0 * m_gridConfig.domainHalfExtent) / static_cast<double>(m_width - 1U);

        // Compute verbatim Pais & Vogel/Willis parameters from isolated Thinker headers.
        const auto& p = m_unifiedParams.pais;
        m_paisMaxPoyntingFlux = thinkers::ComputePaisMaxPoyntingFlux(
            p.geometricFactor, p.surfaceChargeDensity, p.rotationRadius,
            p.angularFrequencyRadS, p.vibrationAmplitude, p.vibrationFrequencyHz, p.translationSpeed
        );
        m_paisInertialMassFactor = thinkers::ComputePaisInertialMassFactor(
            m_paisMaxPoyntingFlux, p.inertialReductionCoupling
        );
        m_paisAmplification = thinkers::ComputePaisAccelAmplification(
            m_paisMaxPoyntingFlux, p.inertialReductionCoupling
        );

        const auto& vw = m_unifiedParams.vogelWillis;
        m_vogelGeomResonance = thinkers::ComputeVogelGeometricResonanceFactor(
            vw.baseTerminationAngleDeg, vw.topTerminationAngleDeg
        );
        m_vogelDrivePower = thinkers::ComputeVogelTorsionDrivePower(
            vw.torsionCoilFrequencyHz, vw.driveVoltageV, m_vogelGeomResonance
        );

        m_totalCrystalPhryll = 0.0;
        for (const auto& anchor : m_stationaryAnchors) {
            m_totalCrystalPhryll += std::abs(anchor.phryllCharge);
        }
        if (m_totalCrystalPhryll <= 0.0) {
            m_totalCrystalPhryll = 1.0;
        }

        m_cumulativeCrystalEnergy = 0.0;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_effectiveCfl = 0.0;

        if (!m_ez.Allocate(context, m_cellCount) ||
            !m_hx.Allocate(context, m_cellCount) ||
            !m_hy.Allocate(context, m_cellCount) ||
            !m_invEps.Allocate(context, m_cellCount) ||
            !m_visualScalar.Allocate(context, m_cellCount)) {
            spdlog::error("FieldGrid2DSolver: Failed allocating FP64 2D grid GPU buffers.");
            Shutdown();
            return false;
        }

        if (!m_ez.ZeroFill(m_stream) ||
            !m_hx.ZeroFill(m_stream) ||
            !m_hy.ZeroFill(m_stream) ||
            !m_visualScalar.ZeroFill(m_stream)) {
            Shutdown();
            return false;
        }

        m_hostVisual.assign(m_cellCount, 0.0);
        m_hostEz.assign(m_cellCount, 0.0);
        m_hostHx.assign(m_cellCount, 0.0);
        m_hostHy.assign(m_cellCount, 0.0);

        // Determine stationary geometry presence from scene anchors.
        bool hasCrystal = false;
        double crystalRadius = 0.45 * m_gridConfig.domainHalfExtent;
        for (const auto& anchor : m_stationaryAnchors) {
            if (anchor.category == "PhryllCrystal") {
                hasCrystal = true;
                if (anchor.radius > 0.0) {
                    crystalRadius = anchor.radius;
                }
            }
        }

        const bool useEssonneRefraction = (m_modelId == scene::ModelId::UnifiedSynthesis);
        const double degToRad = thinkers::constants::kPi / 180.0;
        const dim3 block(16U, 16U);
        const dim3 grid((m_width + block.x - 1U) / block.x, (m_height + block.y - 1U) / block.y);

        if (!m_context->MakeCurrent()) {
            Shutdown();
            return false;
        }

        InitCavityPermittivityKernel<<<grid, block, 0, m_stream>>>(
            m_invEps.Data(), m_width, m_height, m_gridConfig.domainHalfExtent,
            hasCrystal, crystalRadius,
            vw.baseTerminationAngleDeg * degToRad,
            vw.topTerminationAngleDeg * degToRad,
            useEssonneRefraction,
            0.25 * m_gridConfig.domainHalfExtent
        );
        if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
            Shutdown();
            return false;
        }

        if (m_modelId == scene::ModelId::ThinkerEssonne) {
            if (!ComputeSpatialEssonneField()) {
                Shutdown();
                return false;
            }
        }

        if (!Synchronize()) {
            Shutdown();
            return false;
        }

        m_initialized = true;
        spdlog::info(
            "FieldGrid2DSolver initialized [model={}, grid={}x{}, L={:.3g}, dx={:.4g}].",
            scene::ModelIdToString(m_modelId), m_width, m_height, m_gridConfig.domainHalfExtent, m_dx
        );
        return true;
    }

    void FieldGrid2DSolver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_ez.Release();
        m_hx.Release();
        m_hy.Release();
        m_invEps.Release();
        m_visualScalar.Release();
        m_context = nullptr;
        m_stream = nullptr;
        m_initialized = false;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_cumulativeCrystalEnergy = 0.0;
    }

    bool FieldGrid2DSolver::ComputeSpatialEssonneField() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }
        // Derive R and R_s from the first stationary anchor (or default critical R = 1.5 R_s).
        double bodyR = 0.45 * m_gridConfig.domainHalfExtent;
        double Rs = bodyR / 1.5;
        if (!m_stationaryAnchors.empty() && m_stationaryAnchors[0].radius > 0.0) {
            bodyR = m_stationaryAnchors[0].radius;
            const double calcRs = thinkers::ComputeSchwarzschildRadius(
                m_stationaryAnchors[0].mass, m_gravitationalConstant, m_unifiedParams.referenceLightSpeed
            );
            Rs = (calcRs > 0.0) ? calcRs : (bodyR / 1.5);
        }

        const dim3 block(16U, 16U);
        const dim3 grid((m_width + block.x - 1U) / block.x, (m_height + block.y - 1U) / block.y);
        ComputeEssonneVacuumGrid2DKernel<<<grid, block, 0, m_stream>>>(
            m_visualScalar.Data(), m_ez.Data(), m_width, m_height,
            m_gridConfig.domainHalfExtent, bodyR, Rs
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool FieldGrid2DSolver::Step(double dt) {
        if (!m_initialized || m_context == nullptr || !std::isfinite(dt) || dt == 0.0) {
            return false;
        }
        if (!m_context->MakeCurrent()) {
            return false;
        }

        if (m_modelId == scene::ModelId::ThinkerEssonne) {
            if (!ComputeSpatialEssonneField()) {
                return false;
            }
            ++m_stepCount;
            m_simulationTime += dt;
            return true;
        }

        // Enforce strict 2D FDTD Courant-Friedrichs-Lewy (CFL) stability ceiling:
        // S_CFL = c * dt_eff / dx <= 0.45 < 1 / sqrt(2) (proven in FdtdStability.lean).
        const double maxStableDt = (0.45 * m_dx) / m_gridConfig.waveSpeed;
        const double dtEff = std::min(std::abs(dt), maxStableDt);
        const double cfl = (m_gridConfig.waveSpeed * dtEff) / m_dx;
        m_effectiveCfl = cfl;

        // Classify stationary device geometry from anchors.
        int modeKind = 0;
        double cavityRadius = 0.45 * m_gridConfig.domainHalfExtent;
        double coilOffset = 0.55 * m_gridConfig.domainHalfExtent;
        for (const auto& anchor : m_stationaryAnchors) {
            if (anchor.category == "HEEMFG_Cavity" || anchor.category == "HEEMFG_Craft") {
                modeKind = 1;
                if (anchor.radius > 0.0) {
                    cavityRadius = anchor.radius;
                }
            } else if (anchor.category == "PhryllCrystal") {
                modeKind = 2;
            } else if (anchor.category == "TorsionCoil") {
                const double dist = anchor.position.Norm();
                if (dist > 0.0) {
                    coilOffset = dist;
                }
            }
        }

        // Choose physically distinct source amplitudes for Mainstream::MaxwellEM vs Thinkers:
        //   - Mainstream::MaxwellEM : unit baseline drive (1.0), no vacuum polarization / Phryll boost.
        //   - Thinker::Pais         : scaled by Pais vacuum-polarization amplification (1 + kappa * S_max).
        //   - Thinker::VogelWillis  : scaled by 4096 Hz torsion-coil drive power P_drive(f, V, eta_geom).
        //   - Unified::Synthesis    : composes both Pais vacuum polarization and Vogel/Willis drive power.
        double sourceAmp = 1.0;
        if (m_modelId == scene::ModelId::ThinkerPais) {
            sourceAmp = 1.0 * m_paisAmplification;
        } else if (m_modelId == scene::ModelId::ThinkerVogelWillis) {
            sourceAmp = 1.0 * std::max(0.25, m_vogelDrivePower);
        } else if (m_modelId == scene::ModelId::UnifiedSynthesis) {
            sourceAmp = m_paisAmplification * std::max(1.0, m_vogelDrivePower);
        }

        const dim3 block(16U, 16U);
        const dim3 grid((m_width + block.x - 1U) / block.x, (m_height + block.y - 1U) / block.y);

        FdtdUpdateHKernel<<<grid, block, 0, m_stream>>>(
            m_ez.Data(), m_hx.Data(), m_hy.Data(),
            m_width, m_height, cfl, m_gridConfig.dampingCoeff
        );
        if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
            return false;
        }

        FdtdUpdateEzAndVisualKernel<<<grid, block, 0, m_stream>>>(
            m_ez.Data(), m_hx.Data(), m_hy.Data(), m_invEps.Data(), m_visualScalar.Data(),
            m_width, m_height, m_gridConfig.domainHalfExtent,
            cfl, dtEff, m_gridConfig.dampingCoeff,
            m_simulationTime, m_gridConfig.sourceFrequency,
            modeKind, sourceAmp, cavityRadius, coilOffset
        );
        if (!UMGEBUNG_CUDA_CHECK_LAST_ERROR()) {
            return false;
        }

        if (m_modelId == scene::ModelId::ThinkerVogelWillis || m_modelId == scene::ModelId::UnifiedSynthesis) {
            m_cumulativeCrystalEnergy += m_vogelDrivePower * m_totalCrystalPhryll * dtEff;
        }

        ++m_stepCount;
        m_simulationTime += dtEff;
        return true;
    }

    bool FieldGrid2DSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool FieldGrid2DSolver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool FieldGrid2DSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
        // Stationary cavity/crystal anchors do not orbit or translate.
        outBodies = m_stationaryAnchors;
        return m_initialized;
    }

    bool FieldGrid2DSolver::DownloadVisualField() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!m_visualScalar.CopyToHost(m_hostVisual, m_stream) ||
            !m_ez.CopyToHost(m_hostEz, m_stream) ||
            !m_hx.CopyToHost(m_hostHx, m_stream) ||
            !m_hy.CopyToHost(m_hostHy, m_stream)) {
            return false;
        }
        return Synchronize();
    }

    bool FieldGrid2DSolver::FetchFieldGrid2DSnapshot(FieldGrid2DSnapshot& outSnapshot) {
        if (!DownloadVisualField()) {
            return false;
        }

        double minV = m_hostVisual.empty() ? 0.0 : m_hostVisual[0];
        double maxV = minV;
        double sumSq = 0.0;
        double peakAbs = 0.0;

        for (double v : m_hostVisual) {
            minV = std::min(minV, v);
            maxV = std::max(maxV, v);
            sumSq += (v * v);
            peakAbs = std::max(peakAbs, std::abs(v));
        }

        const std::size_t centerIdx = (static_cast<std::size_t>(m_height / 2U) * m_width) + (m_width / 2U);

        outSnapshot = FieldGrid2DSnapshot{};
        outSnapshot.width = m_width;
        outSnapshot.height = m_height;
        outSnapshot.domainHalfExtent = m_gridConfig.domainHalfExtent;
        outSnapshot.scalarField = m_hostVisual;
        outSnapshot.minVal = minV;
        outSnapshot.maxVal = maxV;
        outSnapshot.l2Norm = (m_cellCount > 0U) ? std::sqrt(sumSq / static_cast<double>(m_cellCount)) : 0.0;
        outSnapshot.peakAbsVal = peakAbs;
        outSnapshot.probeSampleValue = (centerIdx < m_hostEz.size()) ? m_hostEz[centerIdx] : 0.0;
        outSnapshot.fieldLabel = (m_modelId == scene::ModelId::ThinkerEssonne)
            ? "Essonne Vacuum Pressure P(x,y)/P_0"
            : m_gridConfig.fieldQuantityLabel;
        return true;
    }

    bool FieldGrid2DSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        FieldGrid2DSnapshot snap{};
        if (!FetchFieldGrid2DSnapshot(snap)) {
            return false;
        }

        double totalEmEnergy = 0.0;
        double sumPressure = 0.0;
        double sumLight = 0.0;
        const double cellArea = m_dx * m_dx;

        for (std::size_t i = 0U; i < m_cellCount; ++i) {
            if (m_modelId == scene::ModelId::ThinkerEssonne) {
                sumPressure += m_hostVisual[i];
                sumLight += m_hostEz[i];
            } else {
                const double ez = m_hostEz[i];
                const double hx = m_hostHx[i];
                const double hy = m_hostHy[i];
                totalEmEnergy += 0.5 * ((ez * ez) + (hx * hx) + (hy * hy)) * cellArea;
            }
        }

        const double invCells = (m_cellCount > 0U) ? (1.0 / static_cast<double>(m_cellCount)) : 0.0;
        const double satRatio = thinkers::ComputeVogelCrystalSaturationRatio(
            m_cumulativeCrystalEnergy, m_unifiedParams.vogelWillis.saturationThreshold
        );

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = m_modelId;
        outDiagnostics.domain = scene::SimulationDomain::FieldGrid2D;
        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.totalEnergy = totalEmEnergy;
        outDiagnostics.fieldEnergy = totalEmEnergy;
        outDiagnostics.fieldL2Norm = snap.l2Norm;
        outDiagnostics.fieldPeakValue = snap.peakAbsVal;
        outDiagnostics.fieldProbeValue = snap.probeSampleValue;
        outDiagnostics.effectiveCflNumber = m_effectiveCfl;

        outDiagnostics.hasUnifiedTelemetry = (m_modelId != scene::ModelId::MainstreamMaxwellEM);
        if (m_modelId == scene::ModelId::ThinkerEssonne) {
            outDiagnostics.meanVacuumPressureRatio = sumPressure * invCells;
            outDiagnostics.meanLocalLightSpeedRatio = sumLight * invCells;
        }
        outDiagnostics.paisMaxPoyntingFlux =
            (m_modelId == scene::ModelId::ThinkerPais || m_modelId == scene::ModelId::UnifiedSynthesis)
                ? m_paisMaxPoyntingFlux : 0.0;
        outDiagnostics.paisInertialMassFactor =
            (m_modelId == scene::ModelId::ThinkerPais || m_modelId == scene::ModelId::UnifiedSynthesis)
                ? m_paisInertialMassFactor : 1.0;
        outDiagnostics.totalPhryllFlux =
            (m_modelId == scene::ModelId::ThinkerVogelWillis || m_modelId == scene::ModelId::UnifiedSynthesis)
                ? ((m_unifiedParams.phryllFieldAmplitude * m_vogelDrivePower) + m_totalCrystalPhryll) : 0.0;
        outDiagnostics.vogelCrystalSaturationRatio =
            (m_modelId == scene::ModelId::ThinkerVogelWillis || m_modelId == scene::ModelId::UnifiedSynthesis)
                ? satRatio : 0.0;
        outDiagnostics.vogelCrystalSinging =
            (m_modelId == scene::ModelId::ThinkerVogelWillis || m_modelId == scene::ModelId::UnifiedSynthesis) &&
            (satRatio >= 1.0);
        outDiagnostics.solverSummary = m_solverName;
        return true;
    }

} // namespace umgebung::models
