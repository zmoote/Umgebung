#include "umgebung/ecs/systems/MicroPhysics.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Umgebung::ecs::systems {

    __global__ void updateParticles(MicroParticle* particles, int numParticles, float dt, float3 gravity) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) return;

        MicroParticle& p = particles[i];

        // Apply Gravity
        p.velocity.x += gravity.x * dt;
        p.velocity.y += gravity.y * dt;
        p.velocity.z += gravity.z * dt;

        // Update Position
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.position.z += p.velocity.z * dt;

        // Simple ground plane collision at y=0
        if (p.position.y < 0.0f) {
            p.position.y = 0.0f;
            p.velocity.y *= -0.5f; // Bounce with damping
        }
    }

    void launchMicroPhysicsKernel(MicroParticle* d_particles, int numParticles, float dt, float3 gravity) {
        int blockSize = 256;
        int numBlocks = (numParticles + blockSize - 1) / blockSize;
        updateParticles<<<numBlocks, blockSize>>>(d_particles, numParticles, dt, gravity);
        cudaDeviceSynchronize();
    }
}
