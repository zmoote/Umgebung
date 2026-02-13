#pragma once

#include <vector_types.h> // For float3

namespace Umgebung::ecs::systems {

    // This struct is now only used for the one-time transfer
    // from ECS to the GPU buffers.
    struct MicroParticle {
        float3 position;
        float3 velocity;
        float mass;
    };

    // Declaration for the CUDA kernel launcher
    void launchMicroPhysicsKernel(float3* positions, float3* velocities, int numParticles, float dt, float3 gravity);

} // namespace Umgebung::ecs::systems
