#include "umgebung/ecs/systems/PhysicsSystem.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/ecs/components/MicroBody.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ecs/systems/ObserverSystem.hpp"
#include "umgebung/renderer/DebugRenderer.hpp"
#include "umgebung/ecs/systems/MicroPhysics.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <random>

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

            PhysicsSystem::PhysicsSystem(ObserverSystem* observerSystem, renderer::DebugRenderer* debugRenderer)
                : observerSystem_(observerSystem)
                , debugRenderer_(debugRenderer)
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
                if (!gPhysics_) {
                     UMGEBUNG_LOG_CRIT("Cannot create world for scale {}: Physics is null!", static_cast<int>(scale));
                     return;
                }

                PhysicsWorld world;
                world.simScale = 1.0f / toleranceLength;

                world.defaultMaterial = gPhysics_->createMaterial(0.5f, 0.5f, 0.6f);
                if (!world.defaultMaterial)
                {
                    UMGEBUNG_LOG_CRIT("createMaterial failed for scale {}", static_cast<int>(scale));
                    return;
                }

                physx::PxSceneDesc sceneDesc(gPhysics_->getTolerancesScale());
                sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f * world.simScale, 0.0f);
                
                unsigned int numCores = std::thread::hardware_concurrency();
                sceneDesc.cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(numCores);
                sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;

                if (gCudaContextManager_)
                {
                    sceneDesc.cudaContextManager = gCudaContextManager_;
                    sceneDesc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
                    sceneDesc.broadPhaseType = physx::PxBroadPhaseType::eGPU;
                }

                world.scene = gPhysics_->createScene(sceneDesc);
                if (!world.scene)
                {
                    UMGEBUNG_LOG_CRIT("createScene failed for scale {}", static_cast<int>(scale));
                    return;
                }

                worlds_[scale] = world;
                UMGEBUNG_LOG_INFO("Created Physics World for Scale {} (Tol: {}). SimScale: {}", 
                    static_cast<int>(scale), toleranceLength, world.simScale);
            }

            void PhysicsSystem::init(GLFWwindow* window)
            {
                UMGEBUNG_LOG_INFO("Initializing PhysicsSystem");

                gFoundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
                if (!gFoundation_)
                {
                    UMGEBUNG_LOG_CRIT("PxCreateFoundation failed!");
                    return;
                }
                UMGEBUNG_LOG_INFO("PhysX Foundation created");

                physx::PxTolerancesScale tolerances;
                tolerances.length = 1.0f;
                tolerances.speed = 10.0f;
                
                gPhysics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation_, tolerances, true, nullptr);
                if (!gPhysics_) {
                    UMGEBUNG_LOG_CRIT("PxCreatePhysics failed!");
                    return;
                }
                
                if (!PxInitExtensions(*gPhysics_, nullptr)) {
                    UMGEBUNG_LOG_CRIT("PxInitExtensions failed!");
                    return;
                }

                glfwMakeContextCurrent(window);
                physx::PxCudaContextManagerDesc cudaContextManagerDesc;
                HWND hwnd = glfwGetWin32Window(window);
                HDC hdc = GetDC(hwnd);
                cudaContextManagerDesc.graphicsDevice = hdc;

                gCudaContextManager_ = PxCreateCudaContextManager(*gFoundation_, cudaContextManagerDesc, PxGetProfilerCallback());
                if (gCudaContextManager_)
                {
                    if (!gCudaContextManager_->contextIsValid())
                    {
                        UMGEBUNG_LOG_WARN("CUDA context invalid. GPU acceleration will be disabled.");
                        gCudaContextManager_->release();
                        gCudaContextManager_ = nullptr;
                    }
                }
                else
                {
                     UMGEBUNG_LOG_WARN("PxCreateCudaContextManager failed. GPU acceleration will be disabled.");
                }

                if (gCudaContextManager_) {
                     UMGEBUNG_LOG_INFO("GPU Acceleration ENABLED for Multi-Scale Physics Prototype.");
                } else {
                     UMGEBUNG_LOG_WARN("GPU Acceleration DISABLED for Multi-Scale Physics Prototype.");
                }

                createWorldForScale(components::ScaleType::Quantum, 1e-9f);
                createWorldForScale(components::ScaleType::Micro, 1e-4f);
                createWorldForScale(components::ScaleType::Human, 1.0f);
                createWorldForScale(components::ScaleType::Planetary, 1e6f);
                createWorldForScale(components::ScaleType::SolarSystem, 1.5e11f);
                createWorldForScale(components::ScaleType::Galactic, 9e20f);
                createWorldForScale(components::ScaleType::ExtraGalactic, 1e23f);
                createWorldForScale(components::ScaleType::Universal, 1e26f);
                createWorldForScale(components::ScaleType::Multiversal, 1e30f);

                particlePosResource_ = debugRenderer_->getParticleCudaResource();
            }

            void PhysicsSystem::update(entt::registry& registry, float dt, const glm::vec3& cameraPosition)
            {
                if (worlds_.empty() || !gPhysics_) return;

                // This is the old logic that was removed
                updateMicroPhysics(registry, dt);

                auto view = registry.view<components::Transform, components::RigidBody>();
                for (auto entity : view)
                {
                    auto& transform = view.get<components::Transform>(entity);
                    auto& rigidBody = view.get<components::RigidBody>(entity);
                    auto* collider = registry.try_get<components::Collider>(entity);
                    
                    components::ScaleType scale = components::ScaleType::Human;
                    if (registry.all_of<components::ScaleComponent>(entity)) {
                        scale = registry.get<components::ScaleComponent>(entity).type;
                    }

                    if (worlds_.find(scale) == worlds_.end()) {
                         continue;
                    }
                    PhysicsWorld& world = worlds_[scale];

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

                    if (rigidBody.runtimeActor && (rigidBody.dirty || (collider && collider->dirty) || typeMismatch || wrongScene)) {
                        if (wrongScene) {
                            UMGEBUNG_LOG_INFO("Migrating entity {} (Scale change)", static_cast<uint32_t>(entity));
                        }
                        physx::PxScene* oldScene = rigidBody.runtimeActor->getScene();
                        if (oldScene) oldScene->removeActor(*rigidBody.runtimeActor);
                        
                        rigidBody.runtimeActor->release();
                        rigidBody.runtimeActor = nullptr;
                        rigidBody.dirty = false;
                        if(collider) collider->dirty = false;
                    }
                    
                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBody::BodyType::Static)
                    {
                        physx::PxTransform currentPxTransform = rigidBody.runtimeActor->getGlobalPose();
                        
                        physx::PxTransform newPxTransform(
                            {transform.position.x * world.simScale, transform.position.y * world.simScale, transform.position.z * world.simScale},
                            {transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w}
                        );
                        
                        bool posChanged = (currentPxTransform.p - newPxTransform.p).magnitudeSquared() > 0.0001f;
                        bool rotChanged = (currentPxTransform.q.dot(newPxTransform.q)) < 0.9999f;

                        if (posChanged || rotChanged) {
                            static_cast<physx::PxRigidStatic*>(rigidBody.runtimeActor)->setGlobalPose(newPxTransform);
                        }
                    }

                    if (!rigidBody.runtimeActor)
                    {
                        if (!collider) continue;

                        physx::PxShape* shape = nullptr;
                        const float MAX_PHYSICS_SIZE = 10000.0f;

                        switch (collider->type)
                        {
                        case components::Collider::ColliderType::Box:
                        {
                            physx::PxVec3 halfExtents(
                                collider->boxSize.x * transform.scale.x * world.simScale,
                                collider->boxSize.y * transform.scale.y * world.simScale,
                                collider->boxSize.z * transform.scale.z * world.simScale
                            );
                            
                            if (halfExtents.x > MAX_PHYSICS_SIZE || halfExtents.y > MAX_PHYSICS_SIZE || halfExtents.z > MAX_PHYSICS_SIZE) {
                                UMGEBUNG_LOG_WARN("Entity {} is too large for Scale {}! Clamping to {}. (Is: {}, {}, {})", 
                                    static_cast<uint32_t>(entity), static_cast<int>(scale), MAX_PHYSICS_SIZE, halfExtents.x, halfExtents.y, halfExtents.z);
                                halfExtents.x = physx::PxMin(halfExtents.x, MAX_PHYSICS_SIZE);
                                halfExtents.y = physx::PxMin(halfExtents.y, MAX_PHYSICS_SIZE);
                                halfExtents.z = physx::PxMin(halfExtents.z, MAX_PHYSICS_SIZE);
                            }

                            halfExtents.x = physx::PxMax(halfExtents.x, 0.001f);
                            halfExtents.y = physx::PxMax(halfExtents.y, 0.001f);
                            halfExtents.z = physx::PxMax(halfExtents.z, 0.001f);
                            shape = gPhysics_->createShape(physx::PxBoxGeometry(halfExtents), *world.defaultMaterial);
                            break;
                        }
                        case components::Collider::ColliderType::Sphere:
                        {
                            float maxScale = physx::PxMax(transform.scale.x, physx::PxMax(transform.scale.y, transform.scale.z));
                            float radius = collider->sphereRadius * maxScale * world.simScale;
                            
                            if (radius > MAX_PHYSICS_SIZE) {
                                UMGEBUNG_LOG_WARN("Entity {} is too large for Scale {}! Clamping radius to {}. (Is: {})", 
                                    static_cast<uint32_t>(entity), static_cast<int>(scale), MAX_PHYSICS_SIZE, radius);
                                radius = MAX_PHYSICS_SIZE;
                            }

                            radius = physx::PxMax(radius, 0.001f);
                            shape = gPhysics_->createShape(physx::PxSphereGeometry(radius), *world.defaultMaterial);
                            break;
                        }
                        }

                        if (!shape) continue;

                        physx::PxTransform pxTransform(
                            physx::PxVec3(transform.position.x * world.simScale, transform.position.y * world.simScale, transform.position.z * world.simScale),
                            physx::PxQuat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
                        );

                        if (rigidBody.type == components::RigidBody::BodyType::Dynamic)
                        {
                            physx::PxRigidDynamic* dynamicActor = gPhysics_->createRigidDynamic(pxTransform);
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
                            physx::PxRigidStatic* staticActor = gPhysics_->createRigidStatic(pxTransform);
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

                components::ScaleType observerScale = observerSystem_->getCurrentScale();
                int observerScaleInt = static_cast<int>(observerScale);

                for (auto& [scale, world] : worlds_) {
                    if (world.scene) {
                        int worldScaleInt = static_cast<int>(scale);
                        if (worldScaleInt >= observerScaleInt - 1 && worldScaleInt <= observerScaleInt + 1) {
                            world.scene->simulate(dt);
                            world.scene->fetchResults(true);
                        }
                    }
                }

                auto transformView = registry.view<components::Transform, components::RigidBody>();
                for (auto entity : transformView)
                {
                    auto& rigidBody = transformView.get<components::RigidBody>(entity);
                    auto& transform = transformView.get<components::Transform>(entity);

                    components::ScaleType scale = components::ScaleType::Human;
                    if (registry.all_of<components::ScaleComponent>(entity)) {
                        scale = registry.get<components::ScaleComponent>(entity).type;
                    }
                    if (worlds_.find(scale) == worlds_.end()) continue;
                    PhysicsWorld& world = worlds_[scale];


                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBody::BodyType::Dynamic)
                    {
                        physx::PxTransform pxTransform = rigidBody.runtimeActor->getGlobalPose();
                        
                        transform.position = glm::vec3(
                            pxTransform.p.x / world.simScale, 
                            pxTransform.p.y / world.simScale, 
                            pxTransform.p.z / world.simScale
                        );
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

                microPhysicsInitialized_ = false;
                particleCount_ = 0;
                if (d_velocities_) {
                    cudaFree(d_velocities_);
                    d_velocities_ = nullptr;
                }
            }

            void PhysicsSystem::cleanup()
            {
                UMGEBUNG_LOG_INFO("Cleaning up PhysicsSystem");

                for (auto& [scale, world] : worlds_) {
                    if (world.scene) world.scene->release();
                    if (world.defaultMaterial) world.defaultMaterial->release();
                }
                worlds_.clear();

                PxCloseExtensions();

                if (gPhysics_) gPhysics_->release();
                if (gCudaContextManager_) gCudaContextManager_->release();
                if (gFoundation_) gFoundation_->release();
                
                gPhysics_ = nullptr;
                gCudaContextManager_ = nullptr;
                gFoundation_ = nullptr;

                if (d_velocities_) {
                    cudaFree(d_velocities_);
                    d_velocities_ = nullptr;
                }
            }

             void PhysicsSystem::initializeMicroPhysics(entt::registry& registry)
            {
                auto group = registry.group<components::MicroBody>(entt::get<components::Transform>);
                particleCount_ = group.size();
                
                if (particleCount_ == 0) return;
                
                debugRenderer_->setParticleCount(particleCount_);

                cudaMalloc(&d_velocities_, particleCount_ * sizeof(float3));

                std::vector<float3> host_positions(particleCount_);
                std::vector<float3> host_velocities(particleCount_);
                
                size_t i = 0;
                for (auto entity : group) {
                    const auto& transform = group.get<components::Transform>(entity);
                    const auto& body = group.get<components::MicroBody>(entity);
                    host_positions[i] = {transform.position.x, transform.position.y, transform.position.z};
                    host_velocities[i] = {body.velocity.x, body.velocity.y, body.velocity.z};
                    i++;
                }

                cudaMemcpy(d_velocities_, host_velocities.data(), particleCount_ * sizeof(float3), cudaMemcpyHostToDevice);

                float3* d_positions = nullptr;
                size_t num_bytes;
                cudaGraphicsMapResources(1, &particlePosResource_, 0);
                cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &num_bytes, particlePosResource_);
                
                cudaMemcpy(d_positions, host_positions.data(), particleCount_ * sizeof(float3), cudaMemcpyHostToDevice);

                cudaGraphicsUnmapResources(1, &particlePosResource_, 0);
                
                UMGEBUNG_LOG_INFO("Initialized micro-physics with {} particles.", particleCount_);
                microPhysicsInitialized_ = true;
            }

            void PhysicsSystem::updateMicroPhysics(entt::registry& registry, float dt)
            {
                if (!microPhysicsInitialized_) {
                    initializeMicroPhysics(registry);
                }

                if (particleCount_ == 0 || !particlePosResource_) return;

                float3* d_positions = nullptr;
                size_t num_bytes;
                
                cudaGraphicsMapResources(1, &particlePosResource_, 0);
                cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &num_bytes, particlePosResource_);

                float3 gravity = {0.0f, -9.81f, 0.0f};
                launchMicroPhysicsKernel(d_positions, d_velocities_, particleCount_, dt, gravity);

                cudaGraphicsUnmapResources(1, &particlePosResource_, 0);
            }

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
