#include "umgebung/ecs/systems/MacroPhysics.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <math_constants.h>

namespace Umgebung::ecs::systems {

    // --- Math Helpers for float3 ---
    __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
    __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
    __device__ inline float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
    __device__ inline float3 operator*(float b, float3 a) { return make_float3(a.x * b, a.y * b, a.z * b); }
    __device__ inline float3 operator/(float3 a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
    
    __device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    __device__ inline float3 cross(float3 a, float3 b) { 
        return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); 
    }
    __device__ inline float lengthSq(float3 a) { return dot(a, a); }
    __device__ inline float length(float3 a) { return sqrtf(lengthSq(a)); }
    __device__ inline float3 normalize(float3 a) { 
        float l = length(a);
        if (l < 1e-6f) return make_float3(0, 0, 0);
        return a / l;
    }

    /**
     * @brief 3D to 1D index using prime coefficients to minimize collisions.
     */
    __device__ uint32_t getHash(int3 cellIdx) {
        return (uint32_t)((cellIdx.x * 73856093) ^ (cellIdx.y * 19349663) ^ (cellIdx.z * 83492791));
    }

    __global__ void calculateCellHashKernel(GPURigidBody* bodies, int numBodies, GridParams params) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numBodies) return;

        float3 pos = bodies[i].position;
        
        // Calculate cell coordinates
        int3 cellIdx;
        cellIdx.x = floorf((pos.x - params.minBounds.x) / params.cellSize);
        cellIdx.y = floorf((pos.y - params.minBounds.y) / params.cellSize);
        cellIdx.z = floorf((pos.z - params.minBounds.z) / params.cellSize);

        // Store the hash for sorting
        bodies[i].cellHash = getHash(cellIdx);
    }

    void launchCalculateCellHashKernel(
        CUdeviceptr bodies,
        int numBodies,
        GridParams params,
        CUstream stream)
    {
        if (numBodies == 0) return;

        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        calculateCellHashKernel<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<GPURigidBody*>(bodies),
            numBodies,
            params
        );
    }

    struct BodyHashComparator {
        __device__ bool operator()(const GPURigidBody& a, const GPURigidBody& b) const {
            return a.cellHash < b.cellHash;
        }
    };

    void launchSortBodies(CUdeviceptr bodies, int numBodies) {
        if (numBodies == 0) return;
        thrust::device_ptr<GPURigidBody> bodies_ptr(reinterpret_cast<GPURigidBody*>(bodies));
        thrust::sort(bodies_ptr, bodies_ptr + numBodies, BodyHashComparator());
    }

    __global__ void buildCellIndicesKernel(
        GPURigidBody* bodies, 
        int numBodies, 
        unsigned int* cellStart, 
        unsigned int* cellEnd, 
        int numBuckets) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numBodies) return;

        uint32_t hash = bodies[i].cellHash % numBuckets;

        if (i == 0 || hash != (bodies[i-1].cellHash % numBuckets)) {
            cellStart[hash] = i;
        }
        if (i == numBodies - 1 || hash != (bodies[i+1].cellHash % numBuckets)) {
            cellEnd[hash] = i + 1;
        }
    }

    void launchBuildCellIndicesKernel(
        CUdeviceptr bodies,
        int numBodies,
        CUdeviceptr cellStart,
        CUdeviceptr cellEnd,
        int numBuckets,
        CUstream stream)
    {
        if (numBodies == 0) return;
        cudaMemsetAsync(reinterpret_cast<void*>(cellStart), 0xFF, numBuckets * sizeof(unsigned int), stream);
        cudaMemsetAsync(reinterpret_cast<void*>(cellEnd), 0xFF, numBuckets * sizeof(unsigned int), stream);

        int blockSize = 256;
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        buildCellIndicesKernel<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<GPURigidBody*>(bodies),
            numBodies,
            reinterpret_cast<unsigned int*>(cellStart),
            reinterpret_cast<unsigned int*>(cellEnd),
            numBuckets
        );
    }

    __device__ void resolveCollision(GPURigidBody& a, GPURigidBody& b, float3 normal, float penetration) {
        // 1. Position correction (to avoid sinking)
        float totalInvMass = a.inverseMass + b.inverseMass;
        if (totalInvMass == 0.0f) return;

        float percent = 0.2f; // Slop
        float slop = 0.01f;
        float3 correction = normal * (fmaxf(penetration - slop, 0.0f) / totalInvMass * percent);
        a.position = a.position + (correction * a.inverseMass);
        b.position = b.position - (correction * b.inverseMass);

        // 2. Impulse resolution (Linear)
        float3 relVel = a.linearVelocity - b.linearVelocity;
        float velAlongNormal = dot(relVel, normal);

        // Don't resolve if velocities are separating
        if (velAlongNormal > 0) return;

        float e = fminf(a.restitution, b.restitution);
        float j = -(1.0f + e) * velAlongNormal;
        j /= totalInvMass;

        float3 impulse = normal * j;
        a.linearVelocity = a.linearVelocity + (impulse * a.inverseMass);
        b.linearVelocity = b.linearVelocity - (impulse * b.inverseMass);

        // Note: Angular impulse would involve cross(r, impulse) and inverseInertia, 
        // omitted here for brevity but the hooks are in the struct.
    }

    __global__ void macroCollisionKernel(
        GPURigidBody* bodies, 
        unsigned int* cellStart, 
        unsigned int* cellEnd, 
        int numBodies, 
        GridParams params, 
        int numBuckets,
        float dt) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= numBodies) return;

        GPURigidBody& bodyA = bodies[i];
        if (bodyA.bodyType == GPUBodyType::Static) return;

        // Get cell coordinates
        int3 cellIdx;
        cellIdx.x = floorf((bodyA.position.x - params.minBounds.x) / params.cellSize);
        cellIdx.y = floorf((bodyA.position.y - params.minBounds.y) / params.cellSize);
        cellIdx.z = floorf((bodyA.position.z - params.minBounds.z) / params.cellSize);

        // Loop through 27 neighboring cells
        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    int3 neighborIdx = make_int3(cellIdx.x + x, cellIdx.y + y, cellIdx.z + z);
                    uint32_t hash = getHash(neighborIdx) % numBuckets;

                    unsigned int start = cellStart[hash];
                    unsigned int end = cellEnd[hash];

                    if (start == 0xFFFFFFFF) continue;

                    for (unsigned int j = start; j < end; j++) {
                        if (i == j) continue; // Skip self

                        GPURigidBody& bodyB = bodies[j];
                        
                        // Simple Sphere-Sphere for now
                        float3 diff = bodyA.position - bodyB.position;
                        float distSq = lengthSq(diff);
                        float sumRadii = bodyA.shape.sphereRadius + bodyB.shape.sphereRadius;

                        if (distSq < sumRadii * sumRadii) {
                            float d = sqrtf(distSq);
                            float3 normal = (d > 1e-6f) ? diff / d : make_float3(0, 1, 0);
                            float penetration = sumRadii - d;
                            resolveCollision(bodyA, bodyB, normal, penetration);
                        }
                    }
                }
            }
        }

        // 3. Simple Integration (Semi-Implicit Euler)
        bodyA.position = bodyA.position + (bodyA.linearVelocity * dt);
    }

    void launchMacroCollisionKernel(
        CUdeviceptr bodies,
        int numBodies,
        CUdeviceptr cellStart,
        CUdeviceptr cellEnd,
        int numBuckets,
        GridParams params,
        float dt,
        CUstream stream)
    {
        if (numBodies == 0) return;

        int blockSize = 64; // Smaller block size due to heavy kernel
        int numBlocks = (numBodies + blockSize - 1) / blockSize;

        macroCollisionKernel<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<GPURigidBody*>(bodies),
            reinterpret_cast<unsigned int*>(cellStart),
            reinterpret_cast<unsigned int*>(cellEnd),
            numBodies,
            params,
            numBuckets,
            dt
        );
    }

} // namespace Umgebung::ecs::systems
