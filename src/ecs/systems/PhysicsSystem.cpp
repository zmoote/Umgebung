#include "umgebung/ecs/systems/PhysicsSystem.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/ecs/components/MicroBody.hpp"
#include "umgebung/util/LogMacros.hpp"

#include <cuda_runtime.h> // Added for CUDA memory management
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
                if (!gPhysics_) {
                     UMGEBUNG_LOG_CRIT("Cannot create world for scale {}: Physics is null!", static_cast<int>(scale));
                     return;
                }

                PhysicsWorld world;
                
                // Calculate simulation scale
                // We want the "typical" object at this scale (size = toleranceLength) to map to 1.0 physics units.
                // So: ECS_Value * simScale = PhysX_Value
                // toleranceLength * simScale = 1.0
                // simScale = 1.0 / toleranceLength
                world.simScale = 1.0f / toleranceLength;

                world.defaultMaterial = gPhysics_->createMaterial(0.5f, 0.5f, 0.6f);
                if (!world.defaultMaterial)
                {
                    UMGEBUNG_LOG_CRIT("createMaterial failed for scale {}", static_cast<int>(scale));
                    return;
                }

                // Use global tolerances (Length 1.0, Speed 10.0) for the scene, 
                // since we are scaling everything to fit this range.
                physx::PxSceneDesc sceneDesc(gPhysics_->getTolerancesScale());
                
                // Gravity must be scaled!
                // 9.81 m/s^2.
                // Distance is scaled by simScale. Time is unscaled.
                // Accel = Dist / Time^2.
                // ScaledAccel = (Dist * simScale) / Time^2 = Accel * simScale.
                sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f * world.simScale, 0.0f);
                
                unsigned int numCores = std::thread::hardware_concurrency();
                sceneDesc.cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(numCores);
                sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;

                // GPU Acceleration
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

                // Create Foundation
                gFoundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
                if (!gFoundation_)
                {
                    UMGEBUNG_LOG_CRIT("PxCreateFoundation failed!");
                    return;
                }
                UMGEBUNG_LOG_INFO("PhysX Foundation created");

                // Create Physics with "Human" tolerances (1.0)
                // All other scales will be mapped to this.
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

                // Create Cuda Context Manager
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

            void PhysicsSystem::update(entt::registry& registry, float dt, const glm::vec3& cameraPosition)
            {
                if (worlds_.empty() || !gPhysics_) return;

                // ---------------------------------------------------------
                // 1. Cross-Scale Coupling: Gravity Transfer
                // ---------------------------------------------------------
                glm::vec3 globalGravityForce(0.0f);
                bool hasGravitySource = false;

                // Find all Planetary bodies to act as gravity sources
                auto planetaryView = registry.view<components::Transform, components::RigidBody, components::ScaleComponent>();
                for (auto entity : planetaryView) {
                    const auto& scaleComp = planetaryView.get<components::ScaleComponent>(entity);
                    if (scaleComp.type == components::ScaleType::Planetary) {
                         const auto& transform = planetaryView.get<components::Transform>(entity);
                         
                         // Convert Planet Position to Meters (Planetary Scale ~ 1e6 meters/unit)
                         float metersPerUnit = 1.0f / worlds_[components::ScaleType::Planetary].simScale;
                         glm::vec3 planetPosMeters = transform.position * metersPerUnit;

                         // Convert Camera Position to Meters (Assuming Camera is in Human Scale for now)
                         glm::vec3 cameraPosMeters = cameraPosition; 

                         glm::vec3 direction = planetPosMeters - cameraPosMeters;
                         float distSq = glm::dot(direction, direction);
                         
                         if (distSq > 0.001f) {
                             glm::vec3 dirNorm = glm::normalize(direction);
                             // Simple Gravity: 9.81 towards the planet center
                             globalGravityForce += dirNorm * 9.81f; 
                             hasGravitySource = true;
                         }
                    }
                }

                // Apply Gravity to Meso/Micro Scenes
                if (hasGravitySource) {
                     std::vector<components::ScaleType> affectedScales = { components::ScaleType::Human, components::ScaleType::Micro };
                     for (auto scale : affectedScales) {
                         if (worlds_.count(scale)) {
                             physx::PxScene* scene = worlds_[scale].scene;
                             float simScale = worlds_[scale].simScale;
                             scene->setGravity(physx::PxVec3(
                                 globalGravityForce.x * simScale, 
                                 globalGravityForce.y * simScale, 
                                 globalGravityForce.z * simScale
                             ));
                         }
                     }
                }

                // ---------------------------------------------------------
                // 2. Origin Shifting
                // ---------------------------------------------------------
                const float SHIFT_THRESHOLD = 10000.0f; // 10km
                if (glm::length(cameraPosition) > SHIFT_THRESHOLD) {
                    UMGEBUNG_LOG_INFO("Camera too far from origin ({}), shifting world...", glm::length(cameraPosition));
                    
                    glm::vec3 shift = -cameraPosition;

                    for (auto& [scale, world] : worlds_) {
                        if (world.scene) {
                            physx::PxVec3 pxShift(
                                shift.x * world.simScale,
                                shift.y * world.simScale,
                                shift.z * world.simScale
                            );
                            world.scene->shiftOrigin(pxShift);
                        }
                    }
                    // Note: Camera reset is required externally!
                }

                // ---------------------------------------------------------
                // 3. CUDA Micro-Scale Solver
                // ---------------------------------------------------------
                updateMicroPhysics(registry, dt);

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
                    
                    // Update static actor's pose if transform was changed in editor (and we are in correct scene)
                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBody::BodyType::Static)
                    {
                        physx::PxTransform currentPxTransform = rigidBody.runtimeActor->getGlobalPose();
                        
                        // Apply SimScale to Position
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

                    // Create actor if it doesn't exist
                    if (!rigidBody.runtimeActor)
                    {
                        if (!collider) continue;

                        physx::PxShape* shape = nullptr;
                        const float MAX_PHYSICS_SIZE = 10000.0f;

                        switch (collider->type)
                        {
                        case components::Collider::ColliderType::Box:
                        {
                            // Apply SimScale to Extents
                            physx::PxVec3 halfExtents(
                                collider->boxSize.x * transform.scale.x * world.simScale,
                                collider->boxSize.y * transform.scale.y * world.simScale,
                                collider->boxSize.z * transform.scale.z * world.simScale
                            );
                            
                            // Sanity Check / Clamping
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
                            // Apply SimScale to Radius
                            float maxScale = physx::PxMax(transform.scale.x, physx::PxMax(transform.scale.y, transform.scale.z));
                            float radius = collider->sphereRadius * maxScale * world.simScale;
                            
                            // Sanity Check / Clamping
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

                        // Apply SimScale to Initial Position
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

                    // Determine Scale (Again, to find correct world)
                    components::ScaleType scale = components::ScaleType::Human;
                    if (registry.all_of<components::ScaleComponent>(entity)) {
                        scale = registry.get<components::ScaleComponent>(entity).type;
                    }
                    if (worlds_.find(scale) == worlds_.end()) continue;
                    PhysicsWorld& world = worlds_[scale];


                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBody::BodyType::Dynamic)
                    {
                        physx::PxTransform pxTransform = rigidBody.runtimeActor->getGlobalPose();
                        
                        // Apply Inverse SimScale to get back to ECS units
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
            }

            void PhysicsSystem::cleanup()
            {
                UMGEBUNG_LOG_INFO("Cleaning up PhysicsSystem");

                for (auto& [scale, world] : worlds_) {
                    if (world.scene) {
                        world.scene->release();
                        world.scene = nullptr;
                    }
                    if (world.defaultMaterial) {
                        world.defaultMaterial->release();
                        world.defaultMaterial = nullptr;
                    }
                }
                worlds_.clear();

                PxCloseExtensions();

                if (gPhysics_) {
                    gPhysics_->release();
                    gPhysics_ = nullptr;
                }

                if (gCudaContextManager_) {
                    gCudaContextManager_->release();
                    gCudaContextManager_ = nullptr;
                }

                if (d_particles_) {
                    cudaFree(d_particles_);
                    d_particles_ = nullptr;
                }

                if (gFoundation_) {
                    gFoundation_->release();
                    gFoundation_ = nullptr;
                }
            }

            std::vector<glm::vec3> PhysicsSystem::getMicroParticles() const
            {
               // Deprecated: Render loop should iterate entities now, but keeping for safety if called
               return {};
            }

            void PhysicsSystem::updateMicroPhysics(entt::registry& registry, float dt)
            {
                auto group = registry.group<components::MicroBody>(entt::get<components::Transform>);
                size_t count = group.size();

                if (count == 0) return;

                // Resize GPU buffer if needed
                if (count > particleBufferSize_) {
                    if (d_particles_) cudaFree(d_particles_);
                    particleBufferSize_ = count + 1024; // Buffer slightly to avoid frequent reallocs
                    if (cudaMalloc(&d_particles_, particleBufferSize_ * sizeof(MicroParticle)) != cudaSuccess) {
                        UMGEBUNG_LOG_ERROR("Failed to allocate CUDA memory for {} micro-bodies", particleBufferSize_);
                        particleBufferSize_ = 0;
                        return;
                    }
                }

                // 1. Gather Data (ECS -> Host Buffer)
                std::vector<MicroParticle> hostParticles(count);
                int idx = 0;
                for (auto entity : group) {
                    const auto& transform = group.get<components::Transform>(entity);
                    const auto& body = group.get<components::MicroBody>(entity);
                    
                    hostParticles[idx].position = { transform.position.x, transform.position.y, transform.position.z };
                    hostParticles[idx].velocity = { body.velocity.x, body.velocity.y, body.velocity.z };
                    hostParticles[idx].mass = body.mass;
                    idx++;
                }

                // 2. Upload (Host -> Device)
                cudaMemcpy(d_particles_, hostParticles.data(), count * sizeof(MicroParticle), cudaMemcpyHostToDevice);

                // 3. Execute Kernel
                 float3 gravity = {0.0f, -9.81f, 0.0f};
                 if (worlds_.count(components::ScaleType::Micro)) {
                       // Use Human gravity for visual test, or Micro gravity if strictly correct
                       // Keeping Human gravity for now as requested by user flow
                 }
                 launchMicroPhysicsKernel(d_particles_, static_cast<int>(count), dt, gravity);

                // 4. Download (Device -> Host)
                cudaMemcpy(hostParticles.data(), d_particles_, count * sizeof(MicroParticle), cudaMemcpyDeviceToHost);

                // 5. Scatter Data (Host Buffer -> ECS)
                idx = 0;
                for (auto entity : group) {
                    auto& transform = group.get<components::Transform>(entity);
                    auto& body = group.get<components::MicroBody>(entity);

                    transform.position.x = hostParticles[idx].position.x;
                    transform.position.y = hostParticles[idx].position.y;
                    transform.position.z = hostParticles[idx].position.z;

                    body.velocity.x = hostParticles[idx].velocity.x;
                    body.velocity.y = hostParticles[idx].velocity.y;
                    body.velocity.z = hostParticles[idx].velocity.z;
                    idx++;
                }
            }

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
