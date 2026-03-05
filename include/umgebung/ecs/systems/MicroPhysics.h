#pragma once

#include <vector_types.h> // For float3
#include <cuda.h>

namespace Umgebung::ecs::systems {

    // This struct is now only used for the one-time transfer
    // from ECS to the GPU buffers.
    struct MicroParticle {
        float3 position;
        float3 velocity;
        float mass;
    };

    // Declaration for the CUDA kernel launcher
    void launchMicroPhysicsKernel(CUdeviceptr positions, CUdeviceptr velocities, CUdeviceptr dts, int numParticles, float3 gravity, CUstream stream);

    // Declaration for the time entanglement kernel launcher
    void launchTimeEntanglementKernel(
        CUdeviceptr entityPositions, 
        CUdeviceptr entityDensities, 
        CUdeviceptr entityMultipliers, 
        CUdeviceptr entityTargetedFlags, 
        CUdeviceptr planetPositions, 
        int numEntities, 
        int numPlanets, 
        float globalDt, 
        CUdeviceptr outSubjectiveDts, 
        CUstream stream);

} // namespace Umgebung::ecs::systems

