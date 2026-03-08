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

    /**
     * @brief Updates the constant memory buffer for planetary gravity sources on the GPU.
     * @param hostPlanets Pointer to host array of planet positions.
     * @param numPlanets Number of planets to copy (capped by MAX_PLANETS).
     * @param stream The CUDA stream to use for the operation.
     */
    void updatePlanetConstantMemory(const float3* hostPlanets, int numPlanets, CUstream stream);

    /**
     * @brief Launches the micro-physics kernel to update particle positions and velocities.
     */
    void launchMicroPhysicsKernel(CUdeviceptr positions, CUdeviceptr velocities, CUdeviceptr dts, int numParticles, float3 gravity, CUstream stream);

    /**
     * @brief Launches the time entanglement kernel to calculate subjective time for entities.
     * This version uses constant memory for planetary positions.
     */
    void launchTimeEntanglementKernel(
        CUdeviceptr entityPositions, 
        CUdeviceptr entityDensities, 
        CUdeviceptr entityMultipliers, 
        CUdeviceptr entityTargetedFlags, 
        int numEntities, 
        int numPlanets, 
        float globalDt, 
        CUdeviceptr outSubjectiveDts, 
        CUstream stream);

    /**
     * @brief Launches a CUDA kernel to perform frustum and distance culling on particles.
     * @param positions Device pointer to particle positions.
     * @param numParticles Total number of particles.
     * @param frustumPlanes Array of 6 frustum planes (normal.x, y, z, distance).
     * @param cameraPos Position of the camera for distance culling.
     * @param maxDist Maximum distance from the camera before culling.
     * @param outIndices Device pointer to the index buffer for visible particles.
     * @param outIndirectCommand Device pointer to the indirect draw command buffer.
     * @param stream The CUDA stream.
     */
    void launchCullingKernel(
        CUdeviceptr positions,
        int numParticles,
        const float4 frustumPlanes[6],
        float3 cameraPos,
        float maxDist,
        CUdeviceptr outIndices,
        CUdeviceptr outAlphas,
        CUdeviceptr outIndirectCommand,
        CUstream stream);

} // namespace Umgebung::ecs::systems
