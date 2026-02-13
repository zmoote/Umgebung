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

// Forward declarations for PhysX classes
namespace physx
{
    class PxFoundation;
    class PxPhysics;
    class PxScene;
    class PxCudaContextManager;
    class PxMaterial;
}

struct GLFWwindow;

// Forward declaration for other systems
namespace Umgebung::ecs::systems {
    class ObserverSystem;
}
namespace Umgebung::renderer {
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

            private:
                physx::PxFoundation* gFoundation_ = nullptr;
                physx::PxPhysics* gPhysics_ = nullptr;
                physx::PxCudaContextManager* gCudaContextManager_ = nullptr;
                
                ObserverSystem* observerSystem_ = nullptr;
                renderer::DebugRenderer* debugRenderer_ = nullptr;

                // --- Micro-Physics (CUDA-GL Interop) ---
                cudaGraphicsResource* particlePosResource_ = nullptr; // From DebugRenderer's VBO
                float3* d_velocities_ = nullptr;       // Velocities stored only on device
                size_t particleCount_ = 0;
                size_t particleCapacity_ = 0;
                bool microPhysicsInitialized_ = false;

                // Map of ScaleType to PhysicsWorld
                std::unordered_map<components::ScaleType, PhysicsWorld> worlds_;

                // Helper to create a world (Scene + Material) for a specific scale
                void createWorldForScale(components::ScaleType scale, float toleranceLength);
                
                // One-time setup to copy initial particle data from ECS to GPU
                void initializeMicroPhysics(entt::registry& registry);

                // Runs the CUDA particle simulation
                void updateMicroPhysics(entt::registry& registry, float dt);
            };

        } // namespace system
    } // namespace ecs
} // namespace Umgebung