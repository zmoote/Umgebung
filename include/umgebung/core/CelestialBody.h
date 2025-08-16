#pragma once
#include "PhysicalObject.h"
#include <PxPhysicsAPI.h>

namespace Umgebung {
    class CelestialBody : public PhysicalObject {
    public:
        CelestialBody(float mass, float radius);
        ~CelestialBody();

        void update(float dt) override;
        void render() const override;

        // Hook for custom physics
        virtual void computeCustomPhysics(float dt);

        // Expose underlying PhysX actor
        physx::PxRigidActor* actor() const { return actor_; }

    private:
        float mass_;
        float radius_;

        physx::PxRigidActor* actor_;   // managed by PhysXManager
        void* customData_;             // device pointer to per?object GPU state
    };
}