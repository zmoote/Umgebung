#pragma once
#include <PxPhysicsAPI.h>

#include <cuda_runtime.h>
#include "scene/Scene.hpp"

namespace umgebung::physics {
    class PhysicsWorld {
    public:
        PhysicsWorld();
        ~PhysicsWorld();
        void update(Scene& scene, float delta_time);

    private:
        physx::PxPhysics* physics_;
        physx::PxScene* scene_;
        physx::PxFoundation* foundation_;
        physx::PxPvd* pvd_;
    };
}