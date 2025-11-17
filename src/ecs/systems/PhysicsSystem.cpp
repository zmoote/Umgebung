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

            

                            // Create CUDA context manager

                            physx::PxCudaContextManagerDesc cudaContextManagerDesc;

            

                            // Get the HWND and HDC for OpenGL interop

                            HWND hwnd = glfwGetWin32Window(window);

                            HDC hdc = GetDC(hwnd);

                            cudaContextManagerDesc.graphicsDevice = hdc;

            

                            gCudaContextManager_ = PxCreateCudaContextManager(*gFoundation_, cudaContextManagerDesc, PxGetProfilerCallback());

                            if (gCudaContextManager_)

                            {

                                if (!gCudaContextManager_->contextIsValid())

                                {

                                    UMGEBUNG_LOG_ERROR("Failed to initialize CUDA context.");

                                    gCudaContextManager_->release();

                                    gCudaContextManager_ = nullptr;

                                }

                                else

                                {

                                    UMGEBUNG_LOG_INFO("PhysX CUDA Context Manager created");

                                }

                            }

                            else

                            {

                                UMGEBUNG_LOG_WARN("PxCreateCudaContextManager failed. Running physics on CPU.");

                            }

            

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

                            

                            if(gCudaContextManager_)

                            {

                                sceneDesc.cudaContextManager = gCudaContextManager_;

                                sceneDesc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;

                                sceneDesc.broadPhaseType = physx::PxBroadPhaseType::eGPU;

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
                if (gScene_) gScene_->release();
                if (gMaterial_) gMaterial_->release();
                if (gCudaContextManager_) gCudaContextManager_->release();
                if (gPhysics_) gPhysics_->release();

                if (gFoundation_) gFoundation_->release();
            }

        } // namespace system
    } // namespace ecs
} // namespace Umgebung
