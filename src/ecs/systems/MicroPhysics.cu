#include "umgebung/ecs/systems/MicroPhysics.h"
#include <device_launch_parameters.h>

namespace Umgebung::ecs::systems {

    __global__ void updateParticles(float3* positions, float3* velocities, float* dts, int numParticles, float3 gravity) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) return;

        float dt = dts[i];

        // Apply Gravity to velocity
        velocities[i].x += gravity.x * dt;
        velocities[i].y += gravity.y * dt;
        velocities[i].z += gravity.z * dt;

        // Update Position using velocity
        positions[i].x += velocities[i].x * dt;
        positions[i].y += velocities[i].y * dt;
        positions[i].z += velocities[i].z * dt;
    }

    void launchMicroPhysicsKernel(CUdeviceptr positions, CUdeviceptr velocities, CUdeviceptr dts, int numParticles, float3 gravity, CUstream stream) {
        if (numParticles == 0) return;
        int blockSize = 256;
        int numBlocks = (numParticles + blockSize - 1) / blockSize;
        updateParticles<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<float3*>(positions), 
            reinterpret_cast<float3*>(velocities), 
            reinterpret_cast<float*>(dts),
            numParticles, 
            gravity);
    }
}
