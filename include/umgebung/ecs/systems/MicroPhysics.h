#pragma once

#include <vector_types.h> 

namespace Umgebung::ecs::systems {

    struct MicroParticle {
        float3 position;
        float3 velocity;
        float mass;
    };

    void launchMicroPhysicsKernel(MicroParticle* d_particles, int numParticles, float dt, float3 gravity);

}
