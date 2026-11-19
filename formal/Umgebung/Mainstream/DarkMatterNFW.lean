import Umgebung.Algebra.ScalarField

/-!
# Umgebung.Mainstream.DarkMatterNFW

Formal specification and **sorry-free** algebraic proofs for Pillar 1's
Navarro-Frenk-White (NFW, ApJ 1996 / 1997) Cold Dark Matter halo baseline
implemented in `include/umgebung/models/mainstream/DarkMatterNFWSolver.hpp`
and `src/models/mainstream/DarkMatterNFWSolver.cu`:

1. `nfwDensity_at_scale_radius`:
   Proves the textbook identity $\rho_{\text{NFW}}(r_s) = \rho_0 / 4$ at
   dimensionless radius $u = r / r_s = 1$.
2. `nfwHaloForce_zero_torque`:
   Proves that the central spherical NFW halo force $\mathbf{F}_{\text{NFW}} =
   -c\,\mathbf{r}$ exerts identically zero torque $\mathbf{r} \times
   \mathbf{F}_{\text{NFW}} = \mathbf{0}$ about the halo center, guaranteeing
   exact orbital angular momentum conservation for every star.
3. `nfwEnclosedMass_linear_in_rho0`:
   Proves that scaling characteristic halo density $\rho_0$ by any scalar
   $\lambda$ scales enclosed halo mass $M_{\text{NFW}}(r)$ by $\lambda$.
-/

namespace Umgebung.Mainstream

open Umgebung.Algebra

section DarkMatterNFWHalo

variable {α : Type u} [F : ScalarField α]

/--
Navarro-Frenk-White (NFW) spherical dark matter density profile at dimensionless
radius `u = r / r_s`:
`ρ_NFW(u) = ρ₀ / (u * (1 + u)²)`
(mirrors `ComputeNFWDensity` in `DarkMatterNFWSolver.hpp`).
-/
def nfwDensity (rho0 u : α) : α :=
  F.div rho0 (F.mul u (F.mul (F.add F.one u) (F.add F.one u)))

/--
**Theorem (NFW Scale-Radius Density Identity).**
At the halo scale radius `r = r_s` (`u = 1`), the NFW density equals exactly
one quarter of the characteristic density:
`ρ_NFW(r_s) = ρ₀ / 4`.
-/
theorem nfwDensity_at_scale_radius (rho0 : α) :
    nfwDensity rho0 F.one = F.div rho0 F.four := by
  unfold nfwDensity
  rw [F.one_mul, ← F.two_def]
  have hTwoSq : F.mul F.two F.two = F.four := by
    rw [F.four_def]
    calc
      F.mul F.two F.two = F.mul F.two (F.add F.one F.one) := by rw [← F.two_def]
      _ = F.add (F.mul F.two F.one) (F.mul F.two F.one) := F.left_distrib _ _ _
      _ = F.add F.two F.two := by rw [F.mul_one]
  rw [hTwoSq]

/--
Inward spherical NFW halo force exerted on a body at displacement `r` relative
to the halo center: `F_NFW(r) = -(radialCoeff) * r`.
-/
def nfwHaloForce (radialCoeff : α) (r : Vec3 α) : Vec3 α :=
  Vec3.smul (F.neg radialCoeff) r

/--
**Theorem (Exact Zero Torque of a Spherical NFW Dark Matter Halo).**
Every body orbiting in the spherical NFW halo potential experiences identically
zero torque about the halo center:
`r × F_NFW(r) = 0`,
establishing exact angular momentum conservation `dL/dt = 0` for
`Mainstream::DarkMatterNFW`.
-/
theorem nfwHaloForce_zero_torque (radialCoeff : α) (r : Vec3 α) :
    Vec3.cross r (nfwHaloForce radialCoeff r) = Vec3.zero := by
  unfold nfwHaloForce
  rw [Vec3.cross_smul_right, Vec3.cross_self, Vec3.smul_zero]

/--
Enclosed spherical NFW mass `M_NFW(r) = fourPi * rho0 * rsCube * f(u)`,
parameterized cleanly by the radial shape factor `f_u = ln(1 + u) - u / (1 + u)`.
-/
def nfwEnclosedMass (fourPi rho0 rsCube shapeFactor : α) : α :=
  F.mul (F.mul (F.mul fourPi rho0) rsCube) shapeFactor

/--
**Theorem (Linearity of Enclosed NFW Halo Mass in Characteristic Density).**
Scaling `ρ₀` by any factor `λ` scales `M_NFW(r)` linearly by `λ`.
-/
theorem nfwEnclosedMass_linear_in_rho0 (fourPi rho0 rsCube shapeFactor lam : α) :
    nfwEnclosedMass fourPi (F.mul lam rho0) rsCube shapeFactor =
      F.mul lam (nfwEnclosedMass fourPi rho0 rsCube shapeFactor) := by
  unfold nfwEnclosedMass
  rw [← F.mul_assoc fourPi lam rho0, F.mul_comm fourPi lam, F.mul_assoc lam fourPi rho0,
      F.mul_assoc lam, F.mul_assoc lam]

end DarkMatterNFWHalo

end Umgebung.Mainstream
