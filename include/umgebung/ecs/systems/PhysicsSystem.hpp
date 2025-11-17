#pragma once

#include <entt/entt.hpp>

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

            class PhysicsSystem
            {
            public:
                PhysicsSystem();
                ~PhysicsSystem();

                void init(GLFWwindow* window);
                void update(entt::registry& registry, float dt);
                void cleanup();

            private:
                physx::PxFoundation* gFoundation_ = nullptr;
                physx::PxPhysics* gPhysics_ = nullptr;
                physx::PxScene* gScene_ = nullptr;
                physx::PxCudaContextManager* gCudaContextManager_ = nullptr;
                physx::PxMaterial* gMaterial_ = nullptr;
            };

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
