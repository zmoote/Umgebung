#include "umgebung/ecs/systems/PhysicsSystem.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/ecs/components/MicroBody.hpp"
#include "umgebung/ecs/components/TimeComponent.hpp"
#include "umgebung/ecs/components/PhryllComponent.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ecs/systems/ObserverSystem.hpp"
#include "umgebung/renderer/DebugRenderer.hpp"
#include "umgebung/ecs/systems/MicroPhysics.h"

#include <glad/glad.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <random>
#include <iterator>
#include <glm/geometric.hpp>

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
                
                // Initialize default grid parameters
                gridParams_.minBounds = {-10000.0f, -10000.0f, -10000.0f};
                gridParams_.cellSize = 10.0f;
                gridParams_.gridResolution = {2000, 2000, 2000};
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
                    else
                    {
                        cuStreamCreate(&gCudaStream_, 0);
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
            }

            void PhysicsSystem::syncParticleResource()
            {
                if (debugRenderer_) {
                    particlePosResource_ = debugRenderer_->getParticleCudaResource();
                    particleIndexResource_ = debugRenderer_->getParticleIndexCudaResource();
                    particleIndirectResource_ = debugRenderer_->getParticleIndirectCudaResource();
                    particleAlphaResource_ = debugRenderer_->getParticleAlphaCudaResource();

                    if (particlePosResource_ && particleIndexResource_ && particleIndirectResource_ && particleAlphaResource_) {
                        UMGEBUNG_LOG_INFO("PhysicsSystem: CUDA Particle resources synchronized successfully (Indirect Rendering & LOD Fading Enabled).");
                    }
                }
            }
            void PhysicsSystem::setCameraFrustum(const renderer::Camera& camera)
            {
                renderer::Camera::Plane planes[6];
                camera.getFrustumPlanes(planes);

                for (int i = 0; i < 6; i++) {
                    frustumPlanes_[i] = {planes[i].normal.x, planes[i].normal.y, planes[i].normal.z, planes[i].distance};
                }
            }

            void PhysicsSystem::update(entt::registry& registry, float dt, const glm::vec3& cameraPosition)
            {
                if (worlds_.empty() || !gPhysics_) return;

                // 1. Calculate Gravity sources (Planets)
                // Optimization: Only update if registry changed or planets moved
                static size_t lastRegistryTick = 0;
                bool planetsMoved = false;
                
                std::vector<glm::vec3> planetaryPositions;
                auto planetView = registry.view<components::Transform, components::ScaleComponent>();
                for (auto entity : planetView) {
                    const auto& scale = planetView.get<components::ScaleComponent>(entity);
                    if (scale.type == components::ScaleType::Planetary) {
                        const auto& transform = planetView.get<components::Transform>(entity);
                        planetaryPositions.push_back(transform.position);
                        // In a more advanced version, we'd check transform.isDirty
                        planetsMoved = true; 
                    }
                }
                
                static size_t lastPlanetaryCount = 0;
                if (planetaryPositions.size() != lastPlanetaryCount) {
                    UMGEBUNG_LOG_TRACE("PhysicsSystem State Change: Found {} planetary gravity sources.", planetaryPositions.size());
                    lastPlanetaryCount = planetaryPositions.size();
                }

                // 2. Update TimeComponent subjectiveDt via CUDA (GPU-Accelerated Entanglement)
                auto timeView = registry.view<components::Transform, ecs::components::TimeComponent>();
                size_t currentTimeEntityCount = std::distance(timeView.begin(), timeView.end());

                if (currentTimeEntityCount > 0) {
                    bool needsFullUpload = false;

                    // Reallocate CUDA buffers if entity count changed
                    if (currentTimeEntityCount != timeEntityCount_) {
                        d_timePositions_.allocate(currentTimeEntityCount);
                        d_timeDensities_.allocate(currentTimeEntityCount);
                        d_timeMultipliers_.allocate(currentTimeEntityCount);
                        d_timeTargetedFlags_.allocate(currentTimeEntityCount);
                        d_subjectiveDts_.allocate(currentTimeEntityCount);

                        timeEntityCount_ = currentTimeEntityCount;

                        host_timePositions_.resize(timeEntityCount_);
                        host_timeDensities_.resize(timeEntityCount_);
                        host_timeMultipliers_.resize(timeEntityCount_);
                        host_timeTargetedFlags_.resize(timeEntityCount_);
                        host_subjectiveDts_.resize(timeEntityCount_);
                        
                        needsFullUpload = true;
                    }

                    // Optimization: Only re-collect and re-upload if needed
                    static bool firstRun = true;
                    if (needsFullUpload || planetsMoved || firstRun) {
                        size_t idx = 0;
                        for (auto entity : timeView) {
                            const auto& transform = timeView.get<components::Transform>(entity);
                            const auto& timeComp = timeView.get<ecs::components::TimeComponent>(entity);
                            host_timePositions_[idx] = {transform.position.x, transform.position.y, transform.position.z};
                            host_timeDensities_[idx] = timeComp.density;
                            host_timeMultipliers_[idx] = timeComp.localTimeMultiplier;
                            host_timeTargetedFlags_[idx] = timeComp.isTargetedByGravity ? 1 : 0;
                            idx++;
                        }

                        d_timePositions_.upload(host_timePositions_);
                        d_timeDensities_.upload(host_timeDensities_);
                        d_timeMultipliers_.upload(host_timeMultipliers_);
                        d_timeTargetedFlags_.upload(host_timeTargetedFlags_);
                        firstRun = false;
                    }

                    // Upload planets if they moved or count changed
                    if (planetaryPositions.size() > 0) {
                        if (planetsMoved || firstRun) {
                            std::vector<float3> host_planets(planetaryPositions.size());
                            for(size_t p=0; p<planetaryPositions.size(); ++p) 
                                host_planets[p] = {planetaryPositions[p].x, planetaryPositions[p].y, planetaryPositions[p].z};
                            
                            updatePlanetConstantMemory(host_planets.data(), (int)host_planets.size(), gCudaStream_);
                        }
                    }

                    // Launch kernel
                    launchTimeEntanglementKernel(
                        d_timePositions_, 
                        d_timeDensities_, 
                        d_timeMultipliers_, 
                        d_timeTargetedFlags_, 
                        (int)timeEntityCount_, 
                        (int)planetaryPositions.size(), 
                        dt, 
                        d_subjectiveDts_, 
                        gCudaStream_);

                    // Download results
                    d_subjectiveDts_.download(host_subjectiveDts_);

                    size_t idx = 0;
                    for (auto entity : timeView) {
                        auto& timeComp = timeView.get<ecs::components::TimeComponent>(entity);
                        timeComp.subjectiveDt = host_subjectiveDts_[idx++];
                    }

                    static int lastTimeUpdateCount = -1;
                    if ((int)timeEntityCount_ != lastTimeUpdateCount) {
                        UMGEBUNG_LOG_TRACE("PhysicsSystem: Using CUDA for time dynamics of {} entities.", timeEntityCount_);
                        lastTimeUpdateCount = (int)timeEntityCount_;
                    }
                }

                components::ScaleType observerScale = observerSystem_->getCurrentScale();
                
                // Update Cross-Scale Proxies (e.g., Human floor in Micro world)
                updateCrossScaleProxies(registry, observerScale);

                // --- 3. Macro Physics Synchronization (GPU Solver) ---
                syncMacroBodies(registry);
                if (macroEntityCount_ > 0) {
                    launchCalculateCellHashKernel(d_macroBodies_, (int)macroEntityCount_, gridParams_, gCudaStream_);
                    
                    // Sort bodies by cell hash for cache locality and fast neighbor lookup
                    launchSortBodies(d_macroBodies_, (int)macroEntityCount_);

                    // Build the lookup table for cell start/end indices
                    launchBuildCellIndicesKernel(
                        d_macroBodies_, 
                        (int)macroEntityCount_, 
                        d_cellStart_, 
                        d_cellEnd_, 
                        numBuckets_, 
                        gCudaStream_);

                    // Narrowphase and impulse resolution
                    launchMacroCollisionKernel(
                        d_macroBodies_,
                        (int)macroEntityCount_,
                        d_cellStart_,
                        d_cellEnd_,
                        numBuckets_,
                        gridParams_,
                        dt,
                        gCudaStream_);

                    // Sync back to ECS
                    downloadMacroBodies(registry);
                }

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

                int observerScaleInt = static_cast<int>(observerScale);

                for (auto& [scale, world] : worlds_) {
                    if (world.scene) {
                        int worldScaleInt = static_cast<int>(scale);
                        
                        bool shouldSimulate = (worldScaleInt >= observerScaleInt - 1 && worldScaleInt <= observerScaleInt + 1);
                        
                        // Force micro simulation if we are human scale and micro objects are present
                        if (scale == components::ScaleType::Micro && observerScale == components::ScaleType::Human) {
                            shouldSimulate = true;
                        }

                        if (shouldSimulate) {
                            if (scale == components::ScaleType::Micro) {
                                updateMicroPhysics(registry, dt, cameraPosition);
                                
                                // Sub-step for stability at extreme scales
                                const int subSteps = 4;
                                float subDt = dt / static_cast<float>(subSteps);
                                for (int i = 0; i < subSteps; ++i) {
                                    world.scene->simulate(subDt);
                                    world.scene->fetchResults(true);
                                }
                            } else {
                                world.scene->simulate(dt);
                                world.scene->fetchResults(true);
                            }
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

                // Clear Proxies first
                for (auto& [entity, scaleMap] : proxies_) {
                    for (auto& [scale, actor] : scaleMap) {
                        if (actor) {
                            physx::PxScene* scene = actor->getScene();
                            if (scene) scene->removeActor(*actor);
                            actor->release();
                        }
                    }
                }
                proxies_.clear();

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
                    if (gCudaContextManager_) gCudaContextManager_->acquireContext();
                    d_velocities_.free();
                    if (gCudaContextManager_) gCudaContextManager_->releaseContext();
                }
            }

            void PhysicsSystem::cleanup()
            {
                UMGEBUNG_LOG_INFO("Cleaning up PhysicsSystem");

                // Cleanup Proxies
                for (auto& [entity, scaleMap] : proxies_) {
                    for (auto& [scale, actor] : scaleMap) {
                        if (actor) {
                            actor->release();
                        }
                    }
                }
                proxies_.clear();

                for (auto& [scale, world] : worlds_) {
                    if (world.scene) world.scene->release();
                    if (world.defaultMaterial) world.defaultMaterial->release();
                }
                worlds_.clear();

                if (gCudaStream_) {
                    cuStreamDestroy(gCudaStream_);
                    gCudaStream_ = 0;
                }

                if (gCudaContextManager_) {
                    gCudaContextManager_->acquireContext();
                }

                d_velocities_.free();
                d_dts_.free();

                d_timePositions_.free();
                d_timeDensities_.free();
                d_timeMultipliers_.free();
                d_timeTargetedFlags_.free();
                d_subjectiveDts_.free();
                
                if (gCudaContextManager_) {
                    gCudaContextManager_->releaseContext();
                }

                if (gCudaContextManager_) {
                    gCudaContextManager_->release();
                    gCudaContextManager_ = nullptr;
                }

                if(gPhysics_)
                {
                    PxCloseExtensions();
                    gPhysics_->release();
                    gPhysics_ = nullptr;
                }

                if (gFoundation_) {
                    gFoundation_->release();
                    gFoundation_ = nullptr;
                }
            }

            void PhysicsSystem::initializeMicroPhysics(entt::registry& registry)
            {
                if (!gCudaContextManager_) return;

                gCudaContextManager_->acquireContext();

                // Exclude entities that have a RigidBody, as they are handled by PhysX
                auto view = registry.view<components::MicroBody, components::Transform>(entt::exclude<components::RigidBody>);
                particleCount_ = 0;
                for (auto entity : view) {
                    particleCount_++;
                }
                
                if (particleCount_ == 0) {
                    gCudaContextManager_->releaseContext();
                    return;
                }
                
                debugRenderer_->setParticleCount(particleCount_);

                d_velocities_.allocate(particleCount_);
                d_dts_.allocate(particleCount_);

                std::vector<float3> host_positions(particleCount_);
                std::vector<float3> host_velocities(particleCount_);
                std::vector<float> host_dts(particleCount_, 0.0f);
                
                size_t i = 0;
                for (auto entity : view) {
                    const auto& transform = view.get<components::Transform>(entity);
                    const auto& body = view.get<components::MicroBody>(entity);
                    host_positions[i] = {transform.position.x, transform.position.y, transform.position.z};
                    host_velocities[i] = {body.velocity.x, body.velocity.y, body.velocity.z};
                    // dt is set each frame dynamically
                    i++;
                }

                d_velocities_.upload(host_velocities);
                d_dts_.upload(host_dts);

                CUdeviceptr d_positions = 0;
                size_t num_bytes;
                CUgraphicsResource resources[] = { (CUgraphicsResource)particlePosResource_ };
                cuGraphicsMapResources(1, resources, gCudaStream_);
                cuGraphicsResourceGetMappedPointer(&d_positions, &num_bytes, resources[0]);
                
                cuMemcpyHtoD(d_positions, host_positions.data(), particleCount_ * sizeof(float3));

                cuGraphicsUnmapResources(1, resources, gCudaStream_);
                
                UMGEBUNG_LOG_INFO("Initialized micro-physics with {} particles.", particleCount_);
                microPhysicsInitialized_ = true;

                gCudaContextManager_->releaseContext();
            }

            void PhysicsSystem::updateMicroPhysics(entt::registry& registry, float dt, const glm::vec3& cameraPosition)
            {
                if (!microPhysicsInitialized_) {
                    initializeMicroPhysics(registry);
                }

                if (particleCount_ == 0 || !particlePosResource_ || !gCudaContextManager_) return;

                gCudaContextManager_->acquireContext();

                CUdeviceptr d_positions = 0;
                size_t num_bytes;
                
                CUgraphicsResource resources[] = { (CUgraphicsResource)particlePosResource_ };
                cuGraphicsMapResources(1, resources, gCudaStream_);
                cuGraphicsResourceGetMappedPointer(&d_positions, &num_bytes, resources[0]);

                float3 gravity = {0.0f, -9.81f, 0.0f};

                // Copy the latest dts to the device
                auto microView = registry.view<components::MicroBody, components::Transform>(entt::exclude<components::RigidBody>);
                std::vector<float> host_dts(particleCount_, 0.0f);
                size_t dti = 0;
                for (auto entity : microView) {
                    if (dti >= particleCount_) break;
                    if (registry.all_of<components::TimeComponent>(entity)) {
                        host_dts[dti] = registry.get<components::TimeComponent>(entity).subjectiveDt;
                    } else {
                        host_dts[dti] = dt; // Default fallback
                    }
                    dti++;
                }
                d_dts_.upload(host_dts);

                // Zero-Copy: Physics kernel writes directly into the OpenGL VBO mapped as d_positions
                launchMicroPhysicsKernel(d_positions, d_velocities_, d_dts_, (int)particleCount_, gravity, gCudaStream_);

                // 2. Frustum & Distance Culling (Indirect Rendering)
                if (particleIndexResource_ && particleIndirectResource_ && particleAlphaResource_) {
                    CUdeviceptr d_indices = 0;
                    CUdeviceptr d_indirect = 0;
                    CUdeviceptr d_alphas = 0;
                    size_t num_bytes_indices, num_bytes_indirect, num_bytes_alphas;

                    CUgraphicsResource cullResources[] = { particleIndexResource_, particleIndirectResource_, particleAlphaResource_ };
                    cuGraphicsMapResources(3, cullResources, gCudaStream_);
                    
                    cuGraphicsResourceGetMappedPointer(&d_indices, &num_bytes_indices, particleIndexResource_);
                    cuGraphicsResourceGetMappedPointer(&d_indirect, &num_bytes_indirect, particleIndirectResource_);
                    cuGraphicsResourceGetMappedPointer(&d_alphas, &num_bytes_alphas, particleAlphaResource_);

                    // We use a cutoff for micro particles
                    float maxDist = 500.0f; 
                    float3 cPos = {cameraPosition.x, cameraPosition.y, cameraPosition.z};

                    launchCullingKernel(
                        d_positions,
                        (int)particleCount_,
                        frustumPlanes_,
                        cPos,
                        maxDist,
                        d_indices,
                        d_alphas,
                        d_indirect,
                        gCudaStream_);

                    cuGraphicsUnmapResources(3, cullResources, gCudaStream_);
                }
                
                cuGraphicsUnmapResources(1, resources, gCudaStream_);
                
                gCudaContextManager_->releaseContext();
            }

            void PhysicsSystem::syncMacroBodies(entt::registry& registry)
            {
                auto view = registry.view<components::Transform, components::RigidBody, components::Collider>();
                
                // Count only manifesting bodies
                size_t manifestingCount = 0;
                for (auto entity : view) {
                    if (registry.all_of<components::PhryllComponent>(entity)) {
                        if (!registry.get<components::PhryllComponent>(entity).isManifesting) continue;
                    }
                    manifestingCount++;
                }

                if (manifestingCount == 0) {
                    macroEntityCount_ = 0;
                    return;
                }

                // If entity count changed significantly, resize buffers
                if (manifestingCount != macroEntityCount_) {
                    macroEntityCount_ = manifestingCount;
                    d_macroBodies_.allocate(macroEntityCount_);
                    
                    numBuckets_ = (int)macroEntityCount_ * 2;
                    d_cellStart_.allocate(numBuckets_);
                    d_cellEnd_.allocate(numBuckets_);
                }

                std::vector<GPURigidBody> hostBodies;
                hostBodies.reserve(macroEntityCount_);

                for (auto entity : view) {
                    // Skip if not manifesting
                    if (registry.all_of<components::PhryllComponent>(entity)) {
                        if (!registry.get<components::PhryllComponent>(entity).isManifesting) continue;
                    }

                    const auto& transform = view.get<components::Transform>(entity);
                    const auto& rb = view.get<components::RigidBody>(entity);
                    const auto& col = view.get<components::Collider>(entity);

                    GPURigidBody gpuBody{};
                    gpuBody.entityID = static_cast<uint32_t>(entity);
                    gpuBody.position = { transform.position.x, transform.position.y, transform.position.z };
                    gpuBody.rotation = { transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w };
                    gpuBody.linearVelocity = { rb.linearVelocity.x, rb.linearVelocity.y, rb.linearVelocity.z };
                    gpuBody.angularVelocity = { rb.angularVelocity.x, rb.angularVelocity.y, rb.angularVelocity.z };
                    gpuBody.mass = rb.mass;
                    gpuBody.restitution = rb.restitution;
                    gpuBody.friction = rb.friction;
                    gpuBody.bodyType = (rb.type == components::RigidBody::BodyType::Static) ? GPUBodyType::Static : GPUBodyType::Dynamic;
                    gpuBody.inverseMass = (gpuBody.bodyType == GPUBodyType::Static) ? 0.0f : (1.0f / rb.mass);

                    // Calculate simplified diagonal inverse inertia tensor
                    if (gpuBody.bodyType == GPUBodyType::Static) {
                        gpuBody.inverseInertia = { 0.0f, 0.0f, 0.0f };
                    } else {
                        if (col.type == components::Collider::ColliderType::Sphere) {
                            gpuBody.colliderType = GPUColliderType::Sphere;
                            float r = col.sphereRadius * (glm::max)(transform.scale.x, (glm::max)(transform.scale.y, transform.scale.z));
                            gpuBody.shape.sphereRadius = r;
                            
                            float i = (2.0f / 5.0f) * rb.mass * r * r;
                            gpuBody.inverseInertia = { 1.0f / i, 1.0f / i, 1.0f / i };
                        } else {
                            gpuBody.colliderType = GPUColliderType::Box;
                            float3 extents = { 
                                col.boxSize.x * transform.scale.x, 
                                col.boxSize.y * transform.scale.y, 
                                col.boxSize.z * transform.scale.z 
                            };
                            gpuBody.shape.boxExtents = extents;
                            
                            // w, h, d
                            float w = extents.x * 2.0f;
                            float h = extents.y * 2.0f;
                            float d = extents.z * 2.0f;
                            
                            float ix = (1.0f / 12.0f) * rb.mass * (h * h + d * d);
                            float iy = (1.0f / 12.0f) * rb.mass * (w * w + d * d);
                            float iz = (1.0f / 12.0f) * rb.mass * (w * w + h * h);
                            gpuBody.inverseInertia = { 1.0f / ix, 1.0f / iy, 1.0f / iz };
                        }
                    }

                    hostBodies.push_back(gpuBody);
                }

                d_macroBodies_.upload(hostBodies);
            }

            void PhysicsSystem::downloadMacroBodies(entt::registry& registry)
            {
                if (macroEntityCount_ == 0) return;

                std::vector<GPURigidBody> hostBodies(macroEntityCount_);
                d_macroBodies_.download(hostBodies);

                for (const auto& gpuBody : hostBodies) {
                    entt::entity entity = static_cast<entt::entity>(gpuBody.entityID);
                    if (registry.valid(entity)) {
                        if (registry.all_of<components::Transform>(entity)) {
                            auto& transform = registry.get<components::Transform>(entity);
                            transform.position = { gpuBody.position.x, gpuBody.position.y, gpuBody.position.z };
                        }
                        if (registry.all_of<components::RigidBody>(entity)) {
                            auto& rb = registry.get<components::RigidBody>(entity);
                            rb.linearVelocity = { gpuBody.linearVelocity.x, gpuBody.linearVelocity.y, gpuBody.linearVelocity.z };
                            rb.angularVelocity = { gpuBody.angularVelocity.x, gpuBody.angularVelocity.y, gpuBody.angularVelocity.z };
                        }
                    }
                }
            }

            void PhysicsSystem::updateCrossScaleProxies(entt::registry& registry, components::ScaleType currentObserverScale)
            {
                // For now, we only implement Human -> Micro proxies for collision
                // If we are at Human or Micro scale, we want Human objects to exist in the Micro scene
                if (currentObserverScale != components::ScaleType::Human && currentObserverScale != components::ScaleType::Micro) return;

                if (worlds_.find(components::ScaleType::Micro) == worlds_.end()) return;
                PhysicsWorld& microWorld = worlds_[components::ScaleType::Micro];

                // Find all Human scale entities with colliders
                auto view = registry.view<components::Transform, components::Collider, components::ScaleComponent>();
                for (auto entity : view) {
                    const auto& transform = view.get<components::Transform>(entity);
                    const auto& collider = view.get<components::Collider>(entity);
                    const auto& scaleComp = view.get<components::ScaleComponent>(entity);

                    if (scaleComp.type != components::ScaleType::Human) continue;

                    // Does it have a proxy in Micro world?
                    physx::PxRigidActor* proxy = nullptr;
                    if (proxies_.count(entity) && proxies_[entity].count(components::ScaleType::Micro)) {
                        proxy = proxies_[entity][components::ScaleType::Micro];
                    }

                    // Create proxy if missing
                    if (!proxy) {
                        physx::PxShape* shape = nullptr;
                        switch (collider->type) {
                            case components::Collider::ColliderType::Box: {
                                physx::PxVec3 halfExtents(
                                    collider->boxSize.x * transform.scale.x * microWorld.simScale,
                                    collider->boxSize.y * transform.scale.y * microWorld.simScale,
                                    collider->boxSize.z * transform.scale.z * microWorld.simScale
                                );
                                shape = gPhysics_->createShape(physx::PxBoxGeometry(halfExtents), *microWorld.defaultMaterial);
                                break;
                            }
                            case components::Collider::ColliderType::Sphere: {
                                float maxScale = (glm::max)(transform.scale.x, (glm::max)(transform.scale.y, transform.scale.z));
                                float radius = collider->sphereRadius * maxScale * microWorld.simScale;
                                shape = gPhysics_->createShape(physx::PxSphereGeometry(radius), *microWorld.defaultMaterial);
                                break;
                            }
                        }

                        if (shape) {
                            physx::PxTransform pxTransform(
                                physx::PxVec3(transform.position.x * microWorld.simScale, transform.position.y * microWorld.simScale, transform.position.z * microWorld.simScale),
                                physx::PxQuat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
                            );
                            
                            physx::PxRigidStatic* staticProxy = gPhysics_->createRigidStatic(pxTransform);
                            staticProxy->attachShape(*shape);
                            microWorld.scene->addActor(*staticProxy);
                            proxy = staticProxy;
                            proxies_[entity][components::ScaleType::Micro] = proxy;
                            shape->release();
                            
                            UMGEBUNG_LOG_INFO("Created Micro Proxy for Human Entity {}", static_cast<uint32_t>(entity));
                        }
                    } else {
                        // Update proxy transform if it's static but might have been moved in editor
                        physx::PxTransform pxTransform(
                            physx::PxVec3(transform.position.x * microWorld.simScale, transform.position.y * microWorld.simScale, transform.position.z * microWorld.simScale),
                            physx::PxQuat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
                        );
                        proxy->setGlobalPose(pxTransform);
                    }
                }
            }

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
