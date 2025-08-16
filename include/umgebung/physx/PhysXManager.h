#pragma once
#include <PxPhysicsAPI.h>
#include <mutex>

namespace Umgebung {
    class PhysXManager {
    public:
        static PhysXManager& instance();
        ~PhysXManager();

        // Scene creation / stepping
        void step(float dt);
        physx::PxPhysics* physics() const { return physics_; }

        // Create a sphere actor (generic for celestial bodies)
        physx::PxRigidActor* createSphere(float mass, float radius,
            const glm::vec3& pos);

    private:
        PhysXManager();   // ctor builds SDK
        void initCuda();  // link CUDA device to PhysX

        physx::PxFoundation* foundation_;
        physx::PxPhysics* physics_;
        physx::PxScene* scene_;
        physx::PxCudaContextManager* cudaCtx_;
        std::mutex mtx_;
    };
}