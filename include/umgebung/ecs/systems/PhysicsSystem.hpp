#pragma once

#include "umgebung/ecs/components/ScaleComponent.hpp"
#include "umgebung/ecs/components/MicroBody.hpp"
#include <entt/entt.hpp>
#include <glm/vec3.hpp>
#include <string>
#include <unordered_map>

// Forward declare for CUDA types
struct cudaGraphicsResource;
struct float3;
struct float4;
#include <cuda.h>

#include "umgebung/util/CudaHelpers.hpp"
#include "umgebung/ecs/systems/MacroPhysics.h"

// Forward declarations for PhysX classes
namespace physx
{
    class PxFoundation;
    class PxPhysics;
    class PxScene;
    class PxCudaContextManager;
    class PxMaterial;
    class PxRigidActor;
}

struct GLFWwindow;

// Forward declaration for other systems
namespace Umgebung::ecs::systems {
    class ObserverSystem;
}
namespace Umgebung::renderer {
    class Camera;
    class DebugRenderer;
}

namespace Umgebung
{
    namespace ecs
    {
        namespace systems
        {

            struct PhysicsWorld {
                physx::PxScene* scene = nullptr;
                physx::PxMaterial* defaultMaterial = nullptr;
                float simScale = 1.0f; // Factor to scale ECS units (meters) to Physics units
            };

            class PhysicsSystem
            {
            public:
                PhysicsSystem(ObserverSystem* observerSystem, renderer::DebugRenderer* debugRenderer);
                ~PhysicsSystem();

                void init(GLFWwindow* window);
                void update(entt::registry& registry, float dt, const glm::vec3& cameraPosition);
                void reset();
                void cleanup();

                void syncParticleResource();

                void setCameraFrustum(const renderer::Camera& camera);

            private:
                physx::PxFoundation* gFoundation_ = nullptr;
                physx::PxPhysics* gPhysics_ = nullptr;
                physx::PxCudaContextManager* gCudaContextManager_ = nullptr;
                CUstream             gCudaStream_ = 0;
                
                ObserverSystem* observerSystem_ = nullptr;
                renderer::DebugRenderer* debugRenderer_ = nullptr;

                // --- Micro-Physics (CUDA-GL Interop) ---
                CUgraphicsResource particlePosResource_ = nullptr; // From DebugRenderer's VBO
                CUgraphicsResource particleIndexResource_ = nullptr; // From DebugRenderer's Index Buffer
                CUgraphicsResource particleIndirectResource_ = nullptr; // From DebugRenderer's Indirect Buffer
                CUgraphicsResource particleAlphaResource_ = nullptr; // From DebugRenderer's Alpha Buffer

                util::DeviceBuffer<float3> d_velocities_;          // Velocities stored only on device
                util::DeviceBuffer<float> d_dts_;                  // Per-particle delta times
                size_t particleCount_ = 0;
                size_t particleCapacity_ = 0;
                bool microPhysicsInitialized_ = false;

                // Frustum planes for culling
                float4 frustumPlanes_[6];

                // --- Time Dynamics (CUDA) ---
                util::DeviceBuffer<float3> d_timePositions_;
                util::DeviceBuffer<float> d_timeDensities_;
                util::DeviceBuffer<float> d_timeMultipliers_;
                util::DeviceBuffer<int> d_timeTargetedFlags_;
                util::DeviceBuffer<float> d_subjectiveDts_;
                size_t timeEntityCount_ = 0;

                // --- Macro Physics (CUDA) ---
                util::DeviceBuffer<GPURigidBody> d_macroBodies_;
                util::DeviceBuffer<unsigned int> d_cellStart_;
                util::DeviceBuffer<unsigned int> d_cellEnd_;
                size_t macroEntityCount_ = 0;
                GridParams gridParams_;
                int numBuckets_ = 0;

                std::vector<float3> host_timePositions_;
                std::vector<float> host_timeDensities_;
                std::vector<float> host_timeMultipliers_;
                std::vector<int> host_timeTargetedFlags_;
                std::vector<float> host_subjectiveDts_;

                // Map of ScaleType to PhysicsWorld
                std::unordered_map<components::ScaleType, PhysicsWorld> worlds_;

                // Helper to create a world (Scene + Material) for a specific scale
                void createWorldForScale(components::ScaleType scale, float toleranceLength);
                
                // One-time setup to copy initial particle data from ECS to GPU
                void initializeMicroPhysics(entt::registry& registry);

                // Runs the CUDA particle simulation
                void updateMicroPhysics(entt::registry& registry, float dt, const glm::vec3& cameraPosition);

                // Synchronizes large bodies (Planets, Stars) to the GPU solver
                void syncMacroBodies(entt::registry& registry);

                // Synchronizes GPU results back to ECS
                void downloadMacroBodies(entt::registry& registry);

                // --- Cross-Scale Proxies ---
                // Maps: [Source Entity] -> [Target Scale] -> [Proxy Actor]
                std::unordered_map<entt::entity, std::unordered_map<components::ScaleType, physx::PxRigidActor*>> proxies_;
                
                void updateCrossScaleProxies(entt::registry& registry, components::ScaleType currentObserverScale);
            };

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
