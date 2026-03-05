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

    __global__ void calculateSubjectiveTime(
        float3* entityPositions, 
        float* entityDensities, 
        float* entityMultipliers, 
        int* entityTargetedFlags, 
        float3* planetPositions, 
        int numEntities, 
        int numPlanets, 
        float globalDt, 
        float* outSubjectiveDts) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numEntities) return;

        if (entityTargetedFlags[i] == 0) {
            outSubjectiveDts[i] = 0.0f;
            return;
        }

        float3 pos = entityPositions[i];
        float maxGravityInfluence = 0.01f;

        for (int p = 0; p < numPlanets; p++) {
            float3 pPos = planetPositions[p];
            float dx = pos.x - pPos.x;
            float dy = pos.y - pPos.y;
            float dz = pos.z - pPos.z;
            float distSq = dx*dx + dy*dy + dz*dz;
            
            float influence = 1.0f / (1.0f + (distSq * 0.0001f));
            if (influence > maxGravityInfluence) maxGravityInfluence = influence;
        }

        float densityMultiplier = 1.0f + ((entityDensities[i] - 3.0f) * 0.2f);
        outSubjectiveDts[i] = globalDt * entityMultipliers[i] * densityMultiplier * maxGravityInfluence;
    }

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
        CUstream stream) 
    {
        if (numEntities == 0) return;
        int blockSize = 256;
        int numBlocks = (numEntities + blockSize - 1) / blockSize;

        calculateSubjectiveTime<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<float3*>(entityPositions),
            reinterpret_cast<float*>(entityDensities),
            reinterpret_cast<float*>(entityMultipliers),
            reinterpret_cast<int*>(entityTargetedFlags),
            reinterpret_cast<float3*>(planetPositions),
            numEntities,
            numPlanets,
            globalDt,
            reinterpret_cast<float*>(outSubjectiveDts)
        );
    }
}
