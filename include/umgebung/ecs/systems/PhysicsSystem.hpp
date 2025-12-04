#pragma once

#include "umgebung/ecs/components/ScaleComponent.hpp"
#include "umgebung/ecs/components/MicroBody.hpp"
#include "umgebung/ecs/systems/MicroPhysics.h" // Contains MicroParticle struct definition
#include <entt/entt.hpp>
#include <glm/vec3.hpp>
#include <string>
#include <unordered_map>

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

// Forward declaration for ObserverSystem
namespace Umgebung::ecs::systems {
    class ObserverSystem;
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
                PhysicsSystem();
                PhysicsSystem(ObserverSystem* observerSystem); // Added constructor
                ~PhysicsSystem();

                void init(GLFWwindow* window);
                void update(entt::registry& registry, float dt, const glm::vec3& cameraPosition);
                void reset();
                void cleanup();

                // Helper to get particle positions for debug rendering
                std::vector<glm::vec3> getMicroParticles() const;

            private:
                physx::PxFoundation* gFoundation_ = nullptr;
                physx::PxPhysics* gPhysics_ = nullptr;
                physx::PxCudaContextManager* gCudaContextManager_ = nullptr;
                
                ObserverSystem* observerSystem_ = nullptr; // Added member

                // Micro-Physics Particles (CUDA)
                MicroParticle* d_particles_ = nullptr;
                size_t particleBufferSize_ = 0; // Track current GPU buffer capacity

                // Map of ScaleType to PhysicsWorld
                std::unordered_map<components::ScaleType, PhysicsWorld> worlds_;

                // Helper to create a world (Scene + Material) for a specific scale
                void createWorldForScale(components::ScaleType scale, float toleranceLength);
                
                // Micro-physics ECS sync
                void updateMicroPhysics(entt::registry& registry, float dt);
            };

        } // namespace system
    } // namespace ecs
} // namespace Umgebung