/**
 * @file Application.cpp
 * @brief Implements the Application class.
 */
#include "umgebung/app/Application.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/asset/ModelLoader.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/ecs/components/MicroBody.hpp"
#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ecs/components/Name.hpp"

#include <GLFW/glfw3.h>
#include <random>
#include <iterator> // Required for std::distance

namespace Umgebung::app {

    Application::Application() = default;

    Application::~Application() {
        shutdown();
    }

    int Application::init() {
        UMGEBUNG_LOG_INFO("Initializing Application Subsystems...");
        window_ = std::make_unique<ui::Window>(1280, 720, "Umgebung");
        if (window_->init() != 0) {
            return -1;
        }

        renderer_ = std::make_unique<renderer::Renderer>();
        renderer_->init();

        // Initialize Editor Camera with default settings (match Renderer's default if possible)
        editorCamera_ = std::make_unique<renderer::Camera>(glm::vec3(0.0f, 0.0f, 3.0f));

        scene_ = std::make_unique<scene::Scene>();
        renderSystem_ = std::make_unique<ecs::systems::RenderSystem>(renderer_.get());
        assetSystem_ = std::make_unique<ecs::systems::AssetSystem>(renderer_->getModelLoader());

        observerSystem_ = std::make_unique<ecs::systems::ObserverSystem>();
        observerSystem_->init();

        // --- New Initialization Order ---
        // 1. Init DebugRenderer
        debugRenderer_ = std::make_unique<renderer::DebugRenderer>();
        debugRenderer_->init();
        
        // 2. Init the particle VBO with a max capacity. This also registers it with CUDA.
        const size_t MAX_PARTICLES = 100000;
        debugRenderer_->initParticles(MAX_PARTICLES);

        // 3. Init PhysicsSystem, passing it the DebugRenderer
        physicsSystem_ = std::make_unique<ecs::systems::PhysicsSystem>(observerSystem_.get(), debugRenderer_.get());
        physicsSystem_->init(window_->getGLFWwindow());

        debugRenderSystem_ = std::make_unique<ecs::systems::DebugRenderSystem>(debugRenderer_.get());

        framebuffer_ = std::make_unique<renderer::Framebuffer>(1280, 720);

        sceneSerializer_ = std::make_unique<scene::SceneSerializer>(scene_.get(), renderer_.get());
        sceneSerializer_->deserialize("assets/scenes/default.umgebung", editorCamera_.get());

        uiManager_ = std::make_unique<ui::UIManager>();
        uiManager_->init(window_->getGLFWwindow(), this, scene_.get(), framebuffer_.get(), renderer_.get(), debugRenderSystem_.get(), sceneSerializer_.get());

        uiManager_->setAppCallback([this]() {
            this->close();
            });

        uiManager_->setStateCallbacks(
            [this]() { this->onPlay(); },
            [this]() { this->onStop(); },
            [this]() { this->onPause(); }
        );

        updateWindowTitle(); // Set initial window title

        UMGEBUNG_LOG_INFO("Application initialized successfully.");
        return 0;
    }

    void Application::updateWindowTitle() {
        std::string title = "Umgebung - ";
        switch (state_) {
            case AppState::Editor:
                title += "Editor";
                break;
            case AppState::Simulate:
                title += "Simulating";
                break;
            case AppState::Paused:
                title += "Paused";
                break;
        }
        window_->setTitle(title);
    }

    void Application::onPlay() {
        if (state_ == AppState::Simulate) return;

        if (state_ == AppState::Editor) {
            sceneSerializer_->serialize("assets/scenes/temp.umgebung", editorCamera_.get());
            UMGEBUNG_LOG_INFO("Simulation Started.");
        } else if (state_ == AppState::Paused) {
             UMGEBUNG_LOG_INFO("Simulation Resumed.");
        }
        
        state_ = AppState::Simulate;
        updateWindowTitle();
    }

    void Application::onStop() {
        if (state_ == AppState::Editor) return;

        state_ = AppState::Editor;
        physicsSystem_->reset();
        sceneSerializer_->deserialize("assets/scenes/temp.umgebung", editorCamera_.get());
        UMGEBUNG_LOG_INFO("Simulation Stopped.");
        updateWindowTitle();
    }

    void Application::onPause() {
        if (state_ == AppState::Editor) return;
        state_ = (state_ == AppState::Simulate) ? AppState::Paused : AppState::Simulate;
        updateWindowTitle();
    }

    void Application::run() {
        UMGEBUNG_LOG_INFO("Entering Main Loop.");
        while (!window_->shouldClose()) {
            window_->beginFrame();

            float currentFrame = static_cast<float>(glfwGetTime());
            deltaTime_ = currentFrame - lastFrame_;
            lastFrame_ = currentFrame;

            uiManager_->beginFrame();
            uiManager_->endFrame();

            processInput(deltaTime_);

            if (auto* viewportPanel = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
                glm::vec2 viewportSize = viewportPanel->getSize();
                if (framebuffer_->getWidth() != viewportSize.x || framebuffer_->getHeight() != viewportSize.y) {
                    if (viewportSize.x > 0 && viewportSize.y > 0) {
                        framebuffer_->resize(viewportSize.x, viewportSize.y);
                        float aspectRatio = viewportSize.x / viewportSize.y;
                        renderer_->getCamera().setPerspective(glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
                        editorCamera_->setPerspective(glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
                    }
                }
            }

            if (state_ == AppState::Simulate) {
                physicsSystem_->update(scene_->getRegistry(), deltaTime_, getActiveCamera().getPosition());
            }

            updateCameraFollow();
            observerSystem_->onUpdate(getActiveCamera(), scene_->getSelectedEntity(), &scene_->getRegistry());

            framebuffer_->bind();
            window_->clear();
            assetSystem_->onUpdate(*scene_);
            
            renderSystem_->onUpdate(*scene_, getActiveCamera(), scene_->getSelectedEntity(), observerSystem_->getCurrentScale());
            
            debugRenderer_->beginFrame(getActiveCamera());
            debugRenderSystem_->onUpdate(scene_->getRegistry());

            debugRenderer_->endFrame();

            framebuffer_->unbind();

            window_->endFrame();
        }
    }

    void Application::close() {
        glfwSetWindowShouldClose(window_->getGLFWwindow(), true);
    }

    void Application::shutdown() {
        if (debugRenderer_) {
            debugRenderer_->shutdown();
        }
        if (uiManager_) {
            uiManager_->shutdown();
        }
    }

    renderer::Camera& Application::getActiveCamera() {
        if (state_ == AppState::Editor) {
            return *editorCamera_;
        }
        return renderer_->getCamera();
    }

    void Application::focusOnEntity(entt::entity entity) {
        auto& registry = scene_->getRegistry();
        if (registry.all_of<ecs::components::Transform>(entity)) {
            auto& transform = registry.get<ecs::components::Transform>(entity);
            
            float distance = 10.0f; 
            
            // Adjust distance based on scale
            if (registry.all_of<ecs::components::ScaleComponent>(entity)) {
                auto& scaleComp = registry.get<ecs::components::ScaleComponent>(entity);
                switch (scaleComp.type) {
                    case ecs::components::ScaleType::Quantum: distance = 0.0001f; break;
                    case ecs::components::ScaleType::Micro:   distance = 0.01f;   break;
                    case ecs::components::ScaleType::Human:   distance = 10.0f;   break;
                    case ecs::components::ScaleType::Planetary: distance = 10000.0f; break;
                    case ecs::components::ScaleType::SolarSystem: distance = 1e8f; break;
                    case ecs::components::ScaleType::Galactic: distance = 1e15f; break;
                    default: distance = 10.0f; break;
                }
            } else {
                // Fallback to transform scale
                distance = glm::max(transform.scale.x, glm::max(transform.scale.y, transform.scale.z)) * 5.0f;
            }

            glm::vec3 targetPos = transform.position;
            glm::vec3 cameraOffset = glm::vec3(0.0f, distance * 0.2f, distance);
            glm::vec3 newCameraPos = targetPos + cameraOffset;

            auto& camera = getActiveCamera();
            camera.setPosition(newCameraPos);
            
            camera.setYaw(-90.0f);
            camera.setPitch(-15.0f); 

            followingEntity_ = entity;
            followOffset_ = cameraOffset;

            UMGEBUNG_LOG_INFO("Focused camera on entity {} at distance {}", static_cast<uint32_t>(entity), distance);
        }
    }

    void Application::updateCameraFollow() {
        if (followingEntity_ == entt::null) return;

        auto& registry = scene_->getRegistry();
        if (!registry.valid(followingEntity_) || !registry.all_of<ecs::components::Transform>(followingEntity_)) {
            followingEntity_ = entt::null;
            return;
        }

        const auto& transform = registry.get<ecs::components::Transform>(followingEntity_);
        auto& camera = getActiveCamera();
        
        // Update camera position to maintain the original relative offset
        camera.setPosition(transform.position + followOffset_);
    }

    void Application::processInput(float deltaTime) {
        GLFWwindow* nativeWindow = window_->getGLFWwindow();

        if (glfwGetKey(nativeWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(nativeWindow, true);
        }

        if (auto* viewport = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
            if (viewport->isFocused() && (glfwGetMouseButton(nativeWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)) {

                glfwSetInputMode(nativeWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

                bool moved = false;
                if (glfwGetKey(nativeWindow, GLFW_KEY_W) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::FORWARD, deltaTime);
                    moved = true;
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_S) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::BACKWARD, deltaTime);
                    moved = true;
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_A) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::LEFT, deltaTime);
                    moved = true;
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_D) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::RIGHT, deltaTime); 
                    moved = true;
                }

                if (moved) {
                    followingEntity_ = entt::null;
                }
                    

                double xpos, ypos;
                glfwGetCursorPos(nativeWindow, &xpos, &ypos);

                if (firstMouse_) {
                    lastX_ = xpos;
                    lastY_ = ypos;
                    firstMouse_ = false;
                }

                double xoffset = xpos - lastX_;
                double yoffset = lastY_ - ypos;

                lastX_ = xpos;
                lastY_ = ypos;

                getActiveCamera().processMouseMovement(xoffset, yoffset);
            }
            else {
                firstMouse_ = true;
                glfwSetInputMode(nativeWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }
    }

}