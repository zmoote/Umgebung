#include "umgebung/ecs/systems/MicroPhysics.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

namespace Umgebung::ecs::systems {

    // Maximum number of planets supported in constant memory
    #define MAX_PLANETS 256
    __constant__ float3 c_planetPositions[MAX_PLANETS];

    /**
     * @brief Optimization: Constant memory is broadcast-optimized for warps.
     * All threads in a warp can read the same planetary coordinate in a single cycle.
     */
    void updatePlanetConstantMemory(const float3* hostPlanets, int numPlanets, CUstream stream) {
        int count = (numPlanets > MAX_PLANETS) ? MAX_PLANETS : numPlanets;
        if (count <= 0) return;
        
        // Note: Using cudaMemcpyToSymbolAsync for better stream integration if needed,
        // but cudaMemcpyToSymbol is generally fine for small constant buffers.
        cudaMemcpyToSymbol(c_planetPositions, hostPlanets, count * sizeof(float3));
    }

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

        // Use constant memory for planets - highly optimized for broadcast
        int planetsToProcess = (numPlanets > MAX_PLANETS) ? MAX_PLANETS : numPlanets;
        for (int p = 0; p < planetsToProcess; p++) {
            float3 pPos = c_planetPositions[p];
            float dx = pos.x - pPos.x;
            float dy = pos.y - pPos.y;
            float dz = pos.z - pPos.z;
            float distSq = dx*dx + dy*dy + dz*dz;
            
            // Proximity-based time dilation (Simplified model)
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
            numEntities,
            numPlanets,
            globalDt,
            reinterpret_cast<float*>(outSubjectiveDts)
        );
    }

    struct DrawElementsIndirectCommand {
        unsigned int count;
        unsigned int instanceCount;
        unsigned int firstIndex;
        int          baseVertex;
        unsigned int baseInstance;
    };

    __global__ void cullParticles(
        float3* positions,
        int numParticles,
        float4* frustumPlanes,
        float3 cameraPos,
        float maxDist,
        unsigned int* outIndices,
        float* outAlphas,
        DrawElementsIndirectCommand* outIndirectCommand)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numParticles) return;

        // Initialize the indirect command (only once)
        if (i == 0) {
            outIndirectCommand->count = 0;
            outIndirectCommand->instanceCount = 1;
            outIndirectCommand->firstIndex = 0;
            outIndirectCommand->baseVertex = 0;
            outIndirectCommand->baseInstance = 0;
        }
        __syncthreads();

        float3 pos = positions[i];
        
        // 1. Distance Culling & Alpha Calculation
        float dx = pos.x - cameraPos.x;
        float dy = pos.y - cameraPos.y;
        float dz = pos.z - cameraPos.z;
        float distSq = dx*dx + dy*dy + dz*dz;
        
        if (distSq > maxDist * maxDist) return;

        // Calculate alpha fade (starts at 80% of maxDist)
        float dist = sqrtf(distSq);
        float startFade = maxDist * 0.8f;
        float alpha = 1.0f;
        if (dist > startFade) {
            alpha = 1.0f - (dist - startFade) / (maxDist - startFade);
        }

        // 2. Frustum Culling (6 planes)
        bool visible = true;
        for (int p = 0; p < 6; p++) {
            float4 plane = frustumPlanes[p];
            if (pos.x * plane.x + pos.y * plane.y + pos.z * plane.z + plane.w < 0.0f) {
                visible = false;
                break;
            }
        }

        if (visible) {
            unsigned int idx = atomicAdd(&(outIndirectCommand->count), 1);
            outIndices[idx] = i;
            outAlphas[idx] = alpha;
        }
    }

    void launchCullingKernel(
        CUdeviceptr positions,
        int numParticles,
        const float4 frustumPlanes[6],
        float3 cameraPos,
        float maxDist,
        CUdeviceptr outIndices,
        CUdeviceptr outAlphas,
        CUdeviceptr outIndirectCommand,
        CUstream stream)
    {
        if (numParticles == 0) return;

        // Upload frustum planes
        float4* d_planes;
        cudaMalloc(&d_planes, 6 * sizeof(float4));
        cudaMemcpyAsync(d_planes, frustumPlanes, 6 * sizeof(float4), cudaMemcpyHostToDevice, stream);

        int blockSize = 256;
        int numBlocks = (numParticles + blockSize - 1) / blockSize;

        cullParticles<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<float3*>(positions),
            numParticles,
            d_planes,
            cameraPos,
            maxDist,
            reinterpret_cast<unsigned int*>(outIndices),
            reinterpret_cast<float*>(outAlphas),
            reinterpret_cast<DrawElementsIndirectCommand*>(outIndirectCommand)
        );

        // Cleanup planes
        cudaFree(d_planes);
    }
}
