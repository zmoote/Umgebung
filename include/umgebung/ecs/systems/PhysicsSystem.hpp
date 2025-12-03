#pragma once

#include "umgebung/ecs/components/ScaleComponent.hpp"
#include <entt/entt.hpp>
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

namespace Umgebung
{
    namespace ecs
    {
        namespace systems
        {

            struct PhysicsWorld {
                physx::PxPhysics* physics = nullptr;
                physx::PxScene* scene = nullptr;
                physx::PxMaterial* defaultMaterial = nullptr;
            };

            class PhysicsSystem
            {
            public:
                PhysicsSystem();
                ~PhysicsSystem();

                void init(GLFWwindow* window);
                void update(entt::registry& registry, float dt);
                void reset();
                void cleanup();

            private:
                physx::PxFoundation* gFoundation_ = nullptr;
                physx::PxCudaContextManager* gCudaContextManager_ = nullptr;
                
                // Map of ScaleType to PhysicsWorld
                std::unordered_map<components::ScaleType, PhysicsWorld> worlds_;

                // Helper to create a world (Physics + Scene + Material) for a specific scale
                void createWorldForScale(components::ScaleType scale, float toleranceLength);
            };

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
