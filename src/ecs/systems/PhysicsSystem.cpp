#include "umgebung/ecs/systems/PhysicsSystem.hpp"
#include "umgebung/ecs/components/RigidBodyComponent.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/util/LogMacros.hpp"

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <windows.h>

#include <PxPhysicsAPI.h>
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

                        void PhysicsSystem::init(GLFWwindow* window)

                        {

                            UMGEBUNG_LOG_INFO("Initializing PhysicsSystem");

            

                            // Create foundation

                            gFoundation_ = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

                            if (!gFoundation_)

                            {

                                UMGEBUNG_LOG_CRIT("PxCreateFoundation failed!");

                                return;

                            }

                            UMGEBUNG_LOG_INFO("PhysX Foundation created");

            

                            // Create physics

                                            gPhysics_ = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation_, physx::PxTolerancesScale(), true, nullptr);

                                            if (!gPhysics_)

                                            {

                                                UMGEBUNG_LOG_CRIT("PxCreatePhysics failed!");

                                                return;

                                            }

                                            UMGEBUNG_LOG_INFO("PhysX Physics created");

                            

                                            if (!PxInitExtensions(*gPhysics_, nullptr))

                                            {

                                                UMGEBUNG_LOG_CRIT("PxInitExtensions failed!");

                                                return;

                                            }

                                            UMGEBUNG_LOG_INFO("PhysX Extensions initialized");

            

                            gMaterial_ = gPhysics_->createMaterial(0.5f, 0.5f, 0.6f);

                            if (!gMaterial_)

                            {

                                UMGEBUNG_LOG_CRIT("createMaterial failed!");

                                return;

                            }

                            UMGEBUNG_LOG_INFO("PhysX Material created");

            

                            // Create scene

                            physx::PxSceneDesc sceneDesc(gPhysics_->getTolerancesScale());

                            sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);

                            sceneDesc.cpuDispatcher = physx::PxDefaultCpuDispatcherCreate(2);

                            sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;

                            // Ensure the GL context is current for CUDA-OpenGL interop
                            glfwMakeContextCurrent(window);

                            physx::PxCudaContextManagerDesc cudaContextManagerDesc;
                            // Use HDC for graphicsDevice as before
                            HWND hwnd = glfwGetWin32Window(window);
                            HDC hdc = GetDC(hwnd);
                            cudaContextManagerDesc.graphicsDevice = hdc;

                            gCudaContextManager_ = PxCreateCudaContextManager(*gFoundation_, cudaContextManagerDesc, PxGetProfilerCallback());
                            if (gCudaContextManager_)
                            {
                                // Check basic validity
                                const bool ctxValid = gCudaContextManager_->contextIsValid() != 0;
                                const bool archOk = gCudaContextManager_->supportsArchSM30() != 0;
                                UMGEBUNG_LOG_INFO("PxCudaContextManager created: contextIsValid={}, supportsArchSM30={}", ctxValid, archOk);

                                if (!ctxValid)
                                {
                                    UMGEBUNG_LOG_WARN("CUDA context manager created but contextIsValid()==false; using CPU physics.");
                                    gCudaContextManager_->release();
                                    gCudaContextManager_ = nullptr;
                                }
                                else if (!archOk)
                                {
                                    UMGEBUNG_LOG_WARN("CUDA device does not support required SM arch; using CPU physics.");
                                    gCudaContextManager_->release();
                                    gCudaContextManager_ = nullptr;
                                }
                                else
                                {
                                    // Ensure PhysX GPU runtime DLL is present before enabling GPU path.
                                    HMODULE physxGpuModule = GetModuleHandleA("PhysXGpu_64.dll");
                                    if (physxGpuModule == NULL)
                                    {
                                        UMGEBUNG_LOG_WARN("PhysX GPU runtime (PhysXGpu_64.dll) not found in process; using CPU physics.");
                                        // keep gCudaContextManager_ for possible later use, but don't enable GPU pipeline
                                    }
                                    else
                                    {
                                        UMGEBUNG_LOG_INFO("PhysX CUDA Context Manager created and PhysXGpu_64.dll present. Enabling GPU pipeline.");
                                        sceneDesc.cudaContextManager = gCudaContextManager_;
                                        sceneDesc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
                                        sceneDesc.broadPhaseType = physx::PxBroadPhaseType::eGPU;
                                    }
                                }
                            }
                            else
                            {
                                UMGEBUNG_LOG_WARN("PxCreateCudaContextManager failed. Running physics on CPU.");
                            }

                            gScene_ = gPhysics_->createScene(sceneDesc);

                            if (!gScene_)

                            {

                                UMGEBUNG_LOG_CRIT("createScene failed!");

                                return;

                            }

                            UMGEBUNG_LOG_INFO("PhysX Scene created");

                        }

            void PhysicsSystem::update(entt::registry& registry, float dt)
            {
                if (!gScene_ || !gPhysics_) return;

                // Create/Update PhysX actors for new RigidBodyComponents
                auto view = registry.view<components::RigidBodyComponent, components::Transform>();
                for (auto entity : view)
                {
                    auto& rigidBody = view.get<components::RigidBodyComponent>(entity);
                    auto& transform = view.get<components::Transform>(entity);

                    if (!rigidBody.runtimeActor)
                    {
                        // Create a box geometry using the entity's scale
                        physx::PxVec3 halfExtents = physx::PxVec3(transform.scale.x * 0.5f, transform.scale.y * 0.5f, transform.scale.z * 0.5f);

                        // GPU-based broadphase does not support zero or negative extents.
                        if (halfExtents.x <= 0.0f || halfExtents.y <= 0.0f || halfExtents.z <= 0.0f)
                        {
                            UMGEBUNG_LOG_WARN("Entity {} has non-positive scale, cannot create valid physics shape. Clamping to a small value.", static_cast<uint32_t>(entity));
                            halfExtents.x = physx::PxMax(halfExtents.x, 0.001f);
                            halfExtents.y = physx::PxMax(halfExtents.y, 0.001f);
                            halfExtents.z = physx::PxMax(halfExtents.z, 0.001f);
                        }

                        physx::PxShape* shape = gPhysics_->createShape(physx::PxBoxGeometry(halfExtents), *gMaterial_);
                        if (!shape)
                        {
                            UMGEBUNG_LOG_ERROR("Failed to create PhysX shape for entity {}", static_cast<uint32_t>(entity));
                            continue;
                        }

                        physx::PxTransform pxTransform(
                            physx::PxVec3(transform.position.x, transform.position.y, transform.position.z),
                            physx::PxQuat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
                        );

                        if (rigidBody.type == components::RigidBodyComponent::BodyType::Dynamic)
                        {
                            physx::PxRigidDynamic* dynamicActor = gPhysics_->createRigidDynamic(pxTransform);
                            if (dynamicActor)
                            {
                                dynamicActor->attachShape(*shape);
                                physx::PxRigidBodyExt::updateMassAndInertia(*dynamicActor, rigidBody.mass);
                                gScene_->addActor(*dynamicActor);
                                rigidBody.runtimeActor = dynamicActor;
                                UMGEBUNG_LOG_INFO("Created dynamic PhysX actor for entity {}", static_cast<uint32_t>(entity));
                            }
                            else
                            {
                                UMGEBUNG_LOG_ERROR("Failed to create dynamic PhysX actor for entity {}", static_cast<uint32_t>(entity));
                            }
                        }
                        else // Static
                        {
                            physx::PxRigidStatic* staticActor = gPhysics_->createRigidStatic(pxTransform);
                            if (staticActor)
                            {
                                staticActor->attachShape(*shape);
                                gScene_->addActor(*staticActor);
                                rigidBody.runtimeActor = staticActor;
                                UMGEBUNG_LOG_INFO("Created static PhysX actor for entity {}", static_cast<uint32_t>(entity));
                            }
                            else
                            {
                                UMGEBUNG_LOG_ERROR("Failed to create static PhysX actor for entity {}", static_cast<uint32_t>(entity));
                            }
                        }
                        shape->release(); // Shape is ref-counted, actor holds a reference
                    }
                }

                // Simulate physics
                gScene_->simulate(dt);
                gScene_->fetchResults(true);

                // Update TransformComponents from PhysX actors
                for (auto entity : view)
                {
                    auto& rigidBody = view.get<components::RigidBodyComponent>(entity);
                    auto& transform = view.get<components::Transform>(entity);

                    if (rigidBody.runtimeActor && rigidBody.type == components::RigidBodyComponent::BodyType::Dynamic)
                    {
                        physx::PxTransform pxTransform = rigidBody.runtimeActor->getGlobalPose();
                        transform.position = glm::vec3(pxTransform.p.x, pxTransform.p.y, pxTransform.p.z);
                        transform.rotation = glm::quat(pxTransform.q.w, pxTransform.q.x, pxTransform.q.y, pxTransform.q.z);
                    }
                }
            }

            void PhysicsSystem::cleanup()
            {
                UMGEBUNG_LOG_INFO("Cleaning up PhysicsSystem");

                // Release scene first
                if (gScene_)
                {
                    UMGEBUNG_LOG_INFO("Releasing PhysX Scene");
                    gScene_->release();
                    gScene_ = nullptr;
                }

                // Release any remaining scene-owned resources (materials, etc.)
                if (gMaterial_)
                {
                    UMGEBUNG_LOG_INFO("Releasing PhysX Material");
                    gMaterial_->release();
                    gMaterial_ = nullptr;
                }

                // Close PhysX extensions before releasing the PxPhysics instance.
                // This matches PxInitExtensions(...) and ensures extension modules drop references to Foundation.
                if (gPhysics_)
                {
                    UMGEBUNG_LOG_INFO("Closing PhysX extensions");
                    PxCloseExtensions();
                }

                // Release PxPhysics instance next
                if (gPhysics_)
                {
                    UMGEBUNG_LOG_INFO("Releasing PxPhysics");
                    gPhysics_->release();
                    gPhysics_ = nullptr;
                }

                // Release CUDA context manager (if any) after releasing PxPhysics / GPU modules.
                // GPU-related modules may keep references to physics internals, so release this later.
                if (gCudaContextManager_)
                {
                    UMGEBUNG_LOG_INFO("Releasing PxCudaContextManager");
                    gCudaContextManager_->release();
                    gCudaContextManager_ = nullptr;
                }

                // Finally release the Foundation
                if (gFoundation_)
                {
                    UMGEBUNG_LOG_INFO("Releasing PxFoundation");
                    gFoundation_->release();
                    gFoundation_ = nullptr;
                }
            }

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
