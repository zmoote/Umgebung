/**
 * @file DarkMatterNFWSolver.hpp
 * @brief Pillar 1 (Mainstream Astrophysics Baseline): Double-precision (FP64)
 *        CUDA Newtonian N-body solver augmented with a Navarro-Frenk-White
 *        (NFW, 1996/1997) spherical Cold Dark Matter (CDM) halo potential.
 *
 * @details
 * Provides the standard $\Lambda$CDM mainstream astrophysical baseline for
 * 3-way spiral galaxy rotation-curve comparisons on a single Scenario file:
 *   - `Mainstream::Newtonian`     (baryonic $1/r^2$ gravity only),
 *   - `Mainstream::DarkMatterNFW` (baryonic $1/r^2$ + NFW CDM dark matter halo),
 *   - `Thinker::Essonne`          (Chris Essonne `10100.pdf` 2D aether vortex without dark matter).
 *
 * Governing equations (Navarro, Frenk & White, ApJ 462:563 (1996) & ApJ 490:493 (1997)):
 *   - **Halo Density Profile:**
 *     $$\rho_{\text{NFW}}(r) = \frac{\rho_0}{\left(\frac{r}{r_s}\right)\left(1 + \frac{r}{r_s}\right)^2}$$
 *   - **Enclosed Spherical Halo Mass:**
 *     $$M_{\text{NFW}}(r) = 4\pi \rho_0 r_s^3 \left[\ln\!\left(1 + \frac{r}{r_s}\right) - \frac{r / r_s}{1 + r / r_s}\right]$$
 *   - **Inward Radial Gravitational Acceleration:**
 *     $$g_{\text{NFW}}(r) = \frac{G M_{\text{NFW}}(r)}{r^2}$$
 *   - **Exact Analytical Halo Potential Energy per Unit Mass:**
 *     $$\Phi_{\text{NFW}}(r) = -4\pi G \rho_0 r_s^3 \frac{\ln(1 + r / r_s)}{r}$$
 *
 * Because the static spherical NFW halo is strictly conservative
 * ($\mathbf{g}_{\text{NFW}} = -\nabla \Phi_{\text{NFW}}$), Kick-Drift-Kick
 * Velocity Verlet symplectically conserves total mechanical energy
 * $E = T + U_{\text{baryon}} + \sum_i m_i \Phi_{\text{NFW}}(\|\mathbf{r}_i - \mathbf{r}_{\text{halo}}\|)$
 * and total angular momentum about the halo center (corresponding formal
 * theorems live in `formal/Umgebung/Mainstream/DarkMatterNFW.lean`).
 *
 * @note Zero GUI dependencies; compiles identically in Desktop GUI and Headless HPC modes.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
    #define UMGEBUNG_NFW_HD __host__ __device__
#else
    #define UMGEBUNG_NFW_HD
#endif

namespace umgebung::models::mainstream {

    /** @brief Mathematical constant $\pi$ in IEEE-754 double precision. */
    inline constexpr double kNfwPi = 3.141592653589793238462643383279502884;

    /**
     * @brief Evaluates the Navarro-Frenk-White (NFW) spherical density
     *        $\rho_{\text{NFW}}(r) = \rho_0 / [(r/r_s)(1 + r/r_s)^2]$.
     *
     * @param r    Radial distance from the halo center ($r > 0$).
     * @param rho0 Characteristic halo density scale $\rho_0 \ge 0$.
     * @param rs   Halo scale radius $r_s > 0$.
     * @return Mass density $\rho_{\text{NFW}}(r)$ in scene mass / length$^3$.
     */
    UMGEBUNG_NFW_HD [[nodiscard]] inline double ComputeNFWDensity(
        double r,
        double rho0,
        double rs
    ) noexcept {
        if (r <= 0.0 || rho0 <= 0.0 || rs <= 0.0) {
            return 0.0;
        }
        const double u = r / rs;
        const double onePlusU = 1.0 + u;
        return rho0 / (u * onePlusU * onePlusU);
    }

    /**
     * @brief Evaluates the enclosed spherical NFW dark matter mass
     *        $M_{\text{NFW}}(r) = 4\pi \rho_0 r_s^3 [\ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]$.
     *
     * @param r    Radial distance from the halo center ($r \ge 0$).
     * @param rho0 Characteristic halo density scale $\rho_0 \ge 0$.
     * @param rs   Halo scale radius $r_s > 0$.
     * @return Enclosed halo mass $M_{\text{NFW}}(r) \ge 0$ in scene mass units.
     */
    UMGEBUNG_NFW_HD [[nodiscard]] inline double ComputeNFWEnclosedMass(
        double r,
        double rho0,
        double rs
    ) noexcept {
        if (r <= 0.0 || rho0 <= 0.0 || rs <= 0.0) {
            return 0.0;
        }
        const double u = r / rs;
        const double massScale = 4.0 * kNfwPi * rho0 * rs * rs * rs;
        // Use log1p(u) for full IEEE-754 accuracy when u << 1.
        const double bracket = log1p(u) - (u / (1.0 + u));
        return massScale * bracket;
    }

    /**
     * @brief Evaluates the inward NFW gravitational acceleration magnitude
     *        $g_{\text{NFW}}(r) = G M_{\text{NFW}}(r) / r^2$.
     *
     * @details
     * For $u = r/r_s < 10^{-5}$, uses the exact Taylor expansion
     * $(\ln(1+u) - u/(1+u))/u^2 = \frac{1}{2} - \frac{2}{3}u + \frac{3}{4}u^2$
     * to avoid catastrophic floating-point cancellation near the halo center.
     *
     * @param r    Radial distance from the halo center ($r \ge 0$).
     * @param G    Gravitational constant $G$.
     * @param rho0 Characteristic halo density scale $\rho_0 \ge 0$.
     * @param rs   Halo scale radius $r_s > 0$.
     * @return Inward radial acceleration magnitude $g_{\text{NFW}}(r) \ge 0$.
     */
    UMGEBUNG_NFW_HD [[nodiscard]] inline double ComputeNFWRadialAccel(
        double r,
        double G,
        double rho0,
        double rs
    ) noexcept {
        if (r <= 0.0 || G <= 0.0 || rho0 <= 0.0 || rs <= 0.0) {
            return 0.0;
        }
        const double u = r / rs;
        if (u < 1.0e-5) {
            // Exact small-u limit: g(r) -> 4*pi*G*rho0*rs * (1/2 - 2u/3 + 3u^2/4).
            const double series = 0.5 - ((2.0 / 3.0) * u) + (0.75 * u * u);
            return 4.0 * kNfwPi * G * rho0 * rs * series;
        }
        const double enclosed = ComputeNFWEnclosedMass(r, rho0, rs);
        return (G * enclosed) / (r * r);
    }

    /**
     * @brief Evaluates the exact conservative NFW halo gravitational potential
     *        $\Phi_{\text{NFW}}(r) = -4\pi G \rho_0 r_s^3 \ln(1 + r/r_s) / r$.
     *
     * @details
     * Satisfies $d\Phi_{\text{NFW}}/dr = +g_{\text{NFW}}(r)$ identically for all
     * $r > 0$, with finite central value $\Phi_{\text{NFW}}(0) = -4\pi G \rho_0 r_s^2$.
     *
     * @param r    Radial distance from the halo center ($r \ge 0$).
     * @param G    Gravitational constant $G$.
     * @param rho0 Characteristic halo density scale $\rho_0 \ge 0$.
     * @param rs   Halo scale radius $r_s > 0$.
     * @return Potential energy per unit mass $\Phi_{\text{NFW}}(r) \le 0$.
     */
    UMGEBUNG_NFW_HD [[nodiscard]] inline double ComputeNFWPotential(
        double r,
        double G,
        double rho0,
        double rs
    ) noexcept {
        if (G <= 0.0 || rho0 <= 0.0 || rs <= 0.0) {
            return 0.0;
        }
        if (r <= 0.0) {
            return -4.0 * kNfwPi * G * rho0 * rs * rs;
        }
        const double u = r / rs;
        const double phi0 = -4.0 * kNfwPi * G * rho0 * rs * rs;
        if (u < 1.0e-6) {
            return phi0 * (1.0 - (0.5 * u) + ((1.0 / 3.0) * u * u));
        }
        return phi0 * (log1p(u) / u);
    }

    /**
     * @class DarkMatterNFWSolver
     * @brief FP64 CUDA Newtonian + NFW Dark Matter Halo solver implementing
     *        `umgebung::models::IPhysicsSolver` (`ModelId::MainstreamDarkMatterNFW`).
     */
    class DarkMatterNFWSolver final : public IPhysicsSolver {
    public:
        static constexpr int kGravityTileSize = 256;

        DarkMatterNFWSolver() = default;
        ~DarkMatterNFWSolver() override;

        DarkMatterNFWSolver(const DarkMatterNFWSolver&) = delete;
        DarkMatterNFWSolver& operator=(const DarkMatterNFWSolver&) = delete;
        DarkMatterNFWSolver(DarkMatterNFWSolver&&) = delete;
        DarkMatterNFWSolver& operator=(DarkMatterNFWSolver&&) = delete;

        void SetStreamIndex(std::size_t streamIndex) noexcept override { m_streamIndex = streamIndex; }

        [[nodiscard]] bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) override;

        void Shutdown() noexcept override;
        [[nodiscard]] bool Step(double dt) override;
        [[nodiscard]] bool StepMultiple(std::uint64_t stepCount, double dt) override;
        [[nodiscard]] bool Synchronize() const noexcept override;
        [[nodiscard]] bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) override;
        [[nodiscard]] bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) override;

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] scene::ModelId GetModelId() const noexcept override {
            return scene::ModelId::MainstreamDarkMatterNFW;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Mainstream::DarkMatterNFW (Newtonian + NFW Halo, FP64 CUDA)";
        }

        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override {
            return scene::PhysicsProfileType::Mainstream;
        }

        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override {
            return m_integrator;
        }

        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_bodyCount; }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

    private:
        [[nodiscard]] bool LaunchNFWAccelerationKernel();
        [[nodiscard]] bool DownloadStateToStaging();

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};

        bool m_initialized{false};
        std::size_t m_bodyCount{0U};
        double m_gravitationalConstant{1.0};
        double m_softeningLength{0.0};
        scene::DarkMatterNFWParams m_nfwParams{};
        scene::Vec3d m_haloCenter{};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        // GPU SoA FP64 Device Buffers
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

        // Reusable Host Staging Buffers
        std::vector<double> m_hostPosX;
        std::vector<double> m_hostPosY;
        std::vector<double> m_hostPosZ;
        std::vector<double> m_hostVelX;
        std::vector<double> m_hostVelY;
        std::vector<double> m_hostVelZ;
        std::vector<double> m_hostMass;
    };

} // namespace umgebung::models::mainstream
