#pragma once
#include <vector_types.h>
#include <cuda.h>
#include <stdint.h>

namespace Umgebung::ecs::systems {

    enum class GPUColliderType : int {
        Box = 0,
        Sphere = 1
    };

    enum class GPUBodyType : int {
        Static = 0,
        Dynamic = 1
    };

    struct GridParams {
        float3 minBounds;
        float cellSize;
        int3 gridResolution;
    };

    // Aligned to 16 bytes for optimal CUDA memory access
    struct alignas(16) GPURigidBody {
        uint32_t entityID;          // Maps back to the ECS entity
        uint32_t cellHash;          // Used for the Spatial Grid Broadphase

        // Transform Data
        float3 position;
        float4 rotation;            // Quaternion (x, y, z, w)
        
        // Kinematic State
        float3 linearVelocity;
        float3 angularVelocity;

        // Physics Properties
        float mass;
        float inverseMass;          // Pre-computed (0.0f for static bodies)
        float restitution;
        float friction;
        
        // Simplified Inertia Tensor (diagonal)
        float3 inverseInertia;      

        GPUBodyType bodyType;
        GPUColliderType colliderType;

        union {
            float3 boxExtents;      // For Box (half-sizes)
            float sphereRadius;     // For Sphere
        } shape;
    };

    /**
     * @brief Launches the spatial hashing kernel to assign bodies to grid cells.
     */
    void launchCalculateCellHashKernel(
        CUdeviceptr bodies,
        int numBodies,
        GridParams params,
        CUstream stream);

    /**
     * @brief Sorts bodies by their cell hash.
     */
    void launchSortBodies(
        CUdeviceptr bodies,
        int numBodies);

    /**
     * @brief Builds the cell start and end index table.
     */
    void launchBuildCellIndicesKernel(
        CUdeviceptr bodies,
        int numBodies,
        CUdeviceptr cellStart,
        CUdeviceptr cellEnd,
        int numBuckets,
        CUstream stream);

    /**
     * @brief Launches the narrowphase and impulse resolution kernel.
     */
    void launchMacroCollisionKernel(
        CUdeviceptr bodies,
        int numBodies,
        CUdeviceptr cellStart,
        CUdeviceptr cellEnd,
        int numBuckets,
        GridParams params,
        float dt,
        CUstream stream);

} // namespace Umgebung::ecs::systems
