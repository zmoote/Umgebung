#include "umgebung/ecs/systems/PhysicsSystem.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/util/LogMacros.hpp"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <windows.h>

#include <physx/PxPhysicsAPI.h>
#include <physx/extensions/PxExtensionsAPI.h>
#include <physx/gpu/PxPhysicsGpu.h>
#include <spdlog/spdlog.h>

namespace Umgebung
{
    namespace ecs
    {
        namespace systems
        {

            // PhysX error callback
            class PxErrorCallback : public physx::PxErrorCallback
            {
            public:
                void reportError(physx::PxErrorCode::Enum code, const char* message, const char* file, int line) override
                {
                    UMGEBUNG_LOG_ERROR("PhysX Error: {} ({}) in {}:{}", message, static_cast<int>(code), file, line);
                }
            };

            static PxErrorCallback gErrorCallback;
            static physx::PxDefaultAllocator gAllocator;

            PhysicsSystem::PhysicsSystem()
            {
                UMGEBUNG_LOG_INFO("PhysicsSystem constructor");
            }

            PhysicsSystem::~PhysicsSystem()
            {
                UMGEBUNG_LOG_INFO("PhysicsSystem destructor");
                cleanup();
            }

            void PhysicsSystem::createWorldForScale(components::ScaleType scale, float toleranceLength)
            {
                PhysicsWorld world;
                
                // Create Foundation PER WORLD
                world.foundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
                if (!world.foundation)
                {
                    UMGEBUNG_LOG_CRIT("PxCreateFoundation failed for scale {}", static_cast<int>(scale));
                    return;
                }

                physx::PxTolerancesScale tolerances;
                tolerances.length = toleranceLength;
                tolerances.speed = toleranceLength * 10.0f; 

                world.physics = PxCreatePhysics(PX_PHYSICS_VERSION, *world.foundation, tolerances, true, nullptr);
                if (!world.physics)
                {
                    UMGEBUNG_LOG_CRIT("PxCreatePhysics failed for scale {}", static_cast<int>(scale));
                    world.foundation->release();
                    return;
                }

                if (!PxInitExtensions(*world.physics, nullptr))
                {
                    UMGEBUNG_LOG_CRIT("PxInitExtensions failed for scale {}", static_cast<int>(scale));
                    world.physics->release();
                    world.foundation->release();
                    return;
                }

                world.defaultMaterial = world.physics->createMaterial(0.5f, 0.5f, 0.6f);
                if (!world.defaultMaterial)
                {
                    UMGEBUNG_LOG_CRIT("createMaterial failed for scale {}", static_cast<int>(scale));
                    return;
                }

                physx::PxSceneDesc sceneDesc(tolerances);
                sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
                unsigned int numCores = std::thread::hardware_concurrency();
                sceneDesc.cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(numCores);
                sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;

                // GPU Acceleration - DISABLED FOR MULTI-SCALE TEST
                // sceneDesc.cudaContextManager = gCudaContextManager_;
                // sceneDesc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
                // sceneDesc.broadPhaseType = physx::PxBroadPhaseType::eGPU;

                world.scene = world.physics->createScene(sceneDesc);
                if (!world.scene)
                {
                    UMGEBUNG_LOG_CRIT("createScene failed for scale {}", static_cast<int>(scale));
                    return;
                }

                worlds_[scale] = world;
                UMGEBUNG_LOG_INFO("Created Physics World for Scale {} with tolerance {}", static_cast<int>(scale), toleranceLength);
            }

            void PhysicsSystem::init(GLFWwindow* window)
            {
                UMGEBUNG_LOG_INFO("Initializing PhysicsSystem");

                // NOTE: GPU Acceleration is temporarily disabled for multi-scale architecture testing.
                // We need to figure out how to share CudaContextManager across multiple Foundations,
                // or if we need multiple Managers.
                
                /*
                // Create a temporary foundation just for CudaContextManager creation? 
                // Or maybe we don't need CudaContextManager yet.
                
                glfwMakeContextCurrent(window);
                // ... setup cuda ...
                */
                UMGEBUNG_LOG_WARN("GPU Acceleration DISABLED for Multi-Scale Physics Prototype.");

                // Initialize Worlds for Scales
                createWorldForScale(components::ScaleType::Quantum, 1e-9f);
                createWorldForScale(components::ScaleType::Micro, 1e-4f);
                createWorldForScale(components::ScaleType::Human, 1.0f);
                createWorldForScale(components::ScaleType::Planetary, 1e6f);     // 1000 km
                createWorldForScale(components::ScaleType::SolarSystem, 1.5e11f); // 1 AU
                createWorldForScale(components::ScaleType::Galactic, 9e20f);     // 100k ly
                createWorldForScale(components::ScaleType::ExtraGalactic, 1e23f);
                createWorldForScale(components::ScaleType::Universal, 1e26f);
                createWorldForScale(components::ScaleType::Multiversal, 1e30f);
            }

            void PhysicsSystem::update(entt::registry& registry, float dt)
            {
                if (worlds_.empty()) return;

                // Sync ECS to PhysX
                auto view = registry.view<components::Transform, components::RigidBody>();
                for (auto entity : view)
                {
                    auto& transform = view.get<components::Transform>(entity);
                    auto& rigidBody = view.get<components::RigidBody>(entity);
                    auto* collider = registry.try_get<components::Collider>(entity);
                    
                    // Determine Scale
                    components::ScaleType scale = components::ScaleType::Human;
                    if (registry.all_of<components::ScaleComponent>(entity)) {
                        scale = registry.get<components::ScaleComponent>(entity).type;
                    }

                    if (worlds_.find(scale) == worlds_.end()) {
                        // Fallback or skip if world for scale doesn't exist
                         continue;
                    }
                    PhysicsWorld& world = worlds_[scale];

                    // Check for Scale Change or mismatched physics/scene
                    bool wrongScene = false;
                    if (rigidBody.runtimeActor) {
                        physx::PxScene* actorScene = rigidBody.runtimeActor->getScene();
                        if (actorScene != world.scene) {
                            wrongScene = true;
                        }
                    }

                    bool isActorDynamic = rigidBody.runtimeActor ? rigidBody.runtimeActor->is<physx::PxRigidDynamic>() : false;
                    bool typeMismatch = rigidBody.runtimeActor &&
                                        ((rigidBody.type == components::RigidBody::BodyType::Dynamic && !isActorDynamic) ||
                                         (rigidBody.type == components::RigidBody::BodyType::Static && isActorDynamic));

                    // Recreate actor if dirty, collider dirty, type mismatch, or WRONG SCENE (Scale change)
                    if (rigidBody.runtimeActor && (rigidBody.dirty || (collider && collider->dirty) || typeMismatch || wrongScene)) {
                        physx::PxScene* oldScene = rigidBody.runtimeActor->getScene();
                        if (oldScene) oldScene->removeActor(*rigidBody.runtimeActor);
                        
                        rigidBody.runtimeActor->release();
                        rigidBody.runtimeActor = nullptr;
                        rigidBody.dirty = false;
                        if(collider) collider->dirty = false;
                    }
                    
                    // Update static actor's pose if transform was changed in editor (and we are in correct scene)
                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBody::BodyType::Static)
                    {
                        physx::PxTransform currentPxTransform = rigidBody.runtimeActor->getGlobalPose();
                        physx::PxTransform newPxTransform(
                            {transform.position.x, transform.position.y, transform.position.z},
                            {transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w}
                        );
                        
                        bool posChanged = currentPxTransform.p.x != newPxTransform.p.x || currentPxTransform.p.y != newPxTransform.p.y || currentPxTransform.p.z != newPxTransform.p.z;
                        bool rotChanged = currentPxTransform.q.x != newPxTransform.q.x || currentPxTransform.q.y != newPxTransform.q.y || currentPxTransform.q.z != newPxTransform.q.z || currentPxTransform.q.w != newPxTransform.q.w;

                        if (posChanged || rotChanged) {
                            static_cast<physx::PxRigidStatic*>(rigidBody.runtimeActor)->setGlobalPose(newPxTransform);
                        }
                    }

                    // Create actor if it doesn't exist
                    if (!rigidBody.runtimeActor)
                    {
                        if (!collider) continue;

                        physx::PxShape* shape = nullptr;
                        
                        switch (collider->type)
                        {
                        case components::Collider::ColliderType::Box:
                        {
                            physx::PxVec3 halfExtents(
                                collider->boxSize.x * transform.scale.x,
                                collider->boxSize.y * transform.scale.y,
                                collider->boxSize.z * transform.scale.z
                            );
                            halfExtents.x = physx::PxMax(halfExtents.x, 0.001f);
                            halfExtents.y = physx::PxMax(halfExtents.y, 0.001f);
                            halfExtents.z = physx::PxMax(halfExtents.z, 0.001f);
                            shape = world.physics->createShape(physx::PxBoxGeometry(halfExtents), *world.defaultMaterial);
                            break;
                        }
                        case components::Collider::ColliderType::Sphere:
                        {
                            float maxScale = physx::PxMax(transform.scale.x, physx::PxMax(transform.scale.y, transform.scale.z));
                            float radius = collider->sphereRadius * maxScale;
                            radius = physx::PxMax(radius, 0.001f);
                            shape = world.physics->createShape(physx::PxSphereGeometry(radius), *world.defaultMaterial);
                            break;
                        }
                        }

                        if (!shape) continue;

                        physx::PxTransform pxTransform(
                            physx::PxVec3(transform.position.x, transform.position.y, transform.position.z),
                            physx::PxQuat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
                        );

                        if (rigidBody.type == components::RigidBody::BodyType::Dynamic)
                        {
                            physx::PxRigidDynamic* dynamicActor = world.physics->createRigidDynamic(pxTransform);
                            if (dynamicActor)
                            {
                                dynamicActor->attachShape(*shape);
                                physx::PxRigidBodyExt::updateMassAndInertia(*dynamicActor, rigidBody.mass);
                                world.scene->addActor(*dynamicActor);
                                rigidBody.runtimeActor = dynamicActor;
                            }
                        }
                        else // Static
                        {
                            physx::PxRigidStatic* staticActor = world.physics->createRigidStatic(pxTransform);
                            if (staticActor)
                            {
                                staticActor->attachShape(*shape);
                                world.scene->addActor(*staticActor);
                                rigidBody.runtimeActor = staticActor;
                            }
                        }
                        shape->release();
                        rigidBody.dirty = false;
                        if (collider) collider->dirty = false;
                    }
                }

                // Simulate physics for ALL worlds
                for (auto& [scale, world] : worlds_) {
                    if (world.scene) {
                        world.scene->simulate(dt);
                        world.scene->fetchResults(true);
                    }
                }

                // Update TransformComponents from PhysX actors
                auto transformView = registry.view<components::Transform, components::RigidBody>();
                for (auto entity : transformView)
                {
                    auto& rigidBody = transformView.get<components::RigidBody>(entity);
                    auto& transform = transformView.get<components::Transform>(entity);

                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBody::BodyType::Dynamic)
                    {
                        physx::PxTransform pxTransform = rigidBody.runtimeActor->getGlobalPose();
                        transform.position = glm::vec3(pxTransform.p.x, pxTransform.p.y, pxTransform.p.z);
                        transform.rotation = glm::quat(pxTransform.q.w, pxTransform.q.x, pxTransform.q.y, pxTransform.q.z);
                    }
                }
            }

            void PhysicsSystem::reset()
            {
                UMGEBUNG_LOG_INFO("Resetting PhysicsSystem (All Worlds)");

                for (auto& [scale, world] : worlds_) {
                    if (!world.scene) continue;

                    world.scene->lockWrite();
                    physx::PxU32 nbActors = world.scene->getNbActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC | physx::PxActorTypeFlag::eRIGID_STATIC);
                    std::vector<physx::PxActor*> actors(nbActors);
                    if (nbActors > 0) {
                         world.scene->getActors(physx::PxActorTypeFlag::eRIGID_DYNAMIC | physx::PxActorTypeFlag::eRIGID_STATIC, actors.data(), nbActors);
                         for (auto actor : actors) {
                            world.scene->removeActor(*actor);
                            actor->release();
                        }
                    }
                    world.scene->unlockWrite();
                }
            }

            void PhysicsSystem::cleanup()
            {
                UMGEBUNG_LOG_INFO("Cleaning up PhysicsSystem");

                // Global close extensions call?
                // Docs say PxCloseExtensions() releases the extensions library.
                // If we called PxInitExtensions multiple times, does it refcount? 
                // Not clear. But typically PxCloseExtensions is global.
                // However, if we have multiple foundations, we might need to be careful.
                // Let's try calling PxCloseExtensions() ONCE at the very end.
                // But wait, PxInitExtensions takes a PxPhysics pointer. 
                // This suggests extensions are attached to the PxPhysics instance.
                // If so, we shouldn't call global PxCloseExtensions if it doesn't take args.
                // PxCloseExtensions() takes NO arguments.
                // This implies it shuts down the *entire* extensions module.
                // This is tricky with multiple foundations.
                // Let's assume we call it once.

                for (auto& [scale, world] : worlds_) {
                    if (world.scene) {
                        world.scene->release();
                        world.scene = nullptr;
                    }
                    if (world.defaultMaterial) {
                        world.defaultMaterial->release();
                        world.defaultMaterial = nullptr;
                    }
                    
                    // We created extensions for EACH physics object.
                    // If PxCloseExtensions is global, we might have an issue.
                    // But we can't really do much else.
                    
                    if (world.physics) {
                        world.physics->release();
                        world.physics = nullptr;
                    }
                    
                    if (world.foundation) {
                        world.foundation->release();
                        world.foundation = nullptr;
                    }
                }
                worlds_.clear();

                // PxCloseExtensions(); 
                // Calling this might crash if we have multiple foundations and it tries to access them?
                // Or maybe it's fine.
                // Given the previous error "Foundation destruction failed due to pending module references",
                // it seems PxInitExtensions DOES create references.
                // We might need to call PxCloseExtensions BEFORE releasing foundations.
                
                // Let's try calling it once here.
                PxCloseExtensions();
            }

        } // namespace system
    } // namespace ecs
} // namespace Umgebung