import Umgebung.Algebra.ScalarField
import Umgebung.Numerical.VelocityVerlet
import Umgebung.Mainstream.NewtonianGravity
import Umgebung.Unified.ThinkersModels

/-!
# Umgebung Formal Verification Root Module (`Umgebung`)

This root library aggregates all three formal verification pillars of the
**Umgebung** Reality Simulation Engine:

1. `Umgebung.Algebra.ScalarField` — Self-contained commutative field and 3D
   Cartesian vector (`Vec3`) algebra over which all physical & numerical proofs
   are parameterized.
2. `Umgebung.Numerical.VelocityVerlet` — Sorry-free proofs of time-reversibility,
   discrete linear momentum conservation, and canonical Jacobian symplecticity
   (`det J = 1`) for the Kick-Drift-Kick Velocity Verlet integrator implemented
   in `src/models/mainstream/NewtonianNBodySolver.cu` and
   `src/models/unified/UnifiedFieldSolver.cu`.
3. `Umgebung.Mainstream.NewtonianGravity` — Formalized Pillar 1 classical
   Newtonian N-body laws (action-reaction antisymmetry, central-force torque
   cancellation, and Kepler circular virial energy).
4. `Umgebung.Unified.ThinkersModels` — Formalized Pillar 2 definitions, explicit
   postulated assumptions, and proven algebraic consequences for Nassim Haramein
   (2013), Chris Essonne (2025), Salvatore Pais (US10144532), and Elena Danaan /
   Dan Willis / Marcel Vogel.
-/
