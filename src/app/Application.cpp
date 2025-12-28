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

        physicsSystem_ = std::make_unique<ecs::systems::PhysicsSystem>(observerSystem_.get());
        physicsSystem_->init(window_->getGLFWwindow());

        debugRenderer_ = std::make_unique<renderer::DebugRenderer>();
        debugRenderer_->init();
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

        //// Spawn Test Micro-Particles (As Entities)

        //int seed = std::chrono::system_clock::now().time_since_epoch().count();

        //UMGEBUNG_LOG_INFO("The seed is: {}", seed);

        //std::mt19937 rng(seed);
        //std::uniform_real_distribution<float> distPos(-10.0f, 10.0f);
        //std::uniform_real_distribution<float> distHeight(5.0f, 20.0f);
        //
        //// Pre-load sphere mesh
        //auto sphereMesh = renderer_->getModelLoader()->loadMesh("assets/models/Sphere.glb");

        //for (int i = 0; i < 1000; ++i) { // Reduced to 1000 for ECS/UI sanity check
        //    auto entity = scene_->createEntity();
        //    scene_->getRegistry().replace<ecs::components::Name>(entity, "Particle_" + std::to_string(i));
        //    auto& transform = scene_->getRegistry().get<ecs::components::Transform>(entity);
        //    transform.position = { distPos(rng), distHeight(rng), distPos(rng) };
        //    transform.scale = { 0.1f, 0.1f, 0.1f }; // Slightly larger for visibility
        //    
        //    scene_->getRegistry().emplace<ecs::components::MicroBody>(entity);
        //    scene_->getRegistry().emplace<ecs::components::ScaleComponent>(entity, ecs::components::ScaleType::Micro);
        //    
        //    // Add Renderable component
        //    ecs::components::Renderable renderable;
        //    renderable.mesh = sphereMesh;
        //    renderable.meshTag = "assets/models/Sphere.glb";
        //    renderable.color = {0.0f, 1.0f, 1.0f, 1.0f}; // Cyan color
        //    scene_->getRegistry().emplace<ecs::components::Renderable>(entity, renderable);
        //}

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
            
            // Toolbar logic (Simple temporary toolbar in the main menu bar or separate window)
            // We can let UIManager handle the drawing, but we need to pass the state controls or expose Application
            // For now, let's draw a simple overlay or just add menu items in UIManager
            // But Application controls the loop.
            // Let's modify UIManager to draw the toolbar.

            uiManager_->endFrame();

            processInput(deltaTime_);

            // Update Observer System to handle camera levels
            observerSystem_->onUpdate(getActiveCamera());

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

            framebuffer_->bind();
            window_->clear();
            assetSystem_->onUpdate(*scene_);
            
            if (state_ == AppState::Simulate) {
                physicsSystem_->update(scene_->getRegistry(), deltaTime_, getActiveCamera().getPosition());
            }
            
            renderSystem_->onUpdate(*scene_, getActiveCamera());
            
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
            
            // Determine an appropriate offset distance
            // Since we normalize physics scales, objects should generally be in the 1-100 unit range.
            // A safe default is often 5-10 units away.
            // We could inspect the collider or mesh bounds later for better framing.
            float distance = 10.0f; 

            // For points (high scale), we might want to be closer or further depending on point size visual.
            // But physically they are small. 
            // Let's just use a fixed offset for now.
            
            glm::vec3 targetPos = transform.position;
            glm::vec3 cameraOffset = glm::vec3(0.0f, 2.0f, distance);
            glm::vec3 newCameraPos = targetPos + cameraOffset;

            // Update Camera
            // We want to look at targetPos.
            // Camera is at newCameraPos.
            // Direction = normalize(targetPos - newCameraPos)
            // We can set position and then hardcode rotation for this offset
            // Offset (0, 2, 10) -> Look down vector (0, -2, -10)
            
            auto& camera = getActiveCamera();
            camera.setPosition(newCameraPos);
            
            // Hardcoded orientation for (0, 2, 10) offset to look at (0,0,0) relative
            // Pitch: atan(-2/10) ~= -11 degrees
            // Yaw: -90 degrees (looking down -Z)
            camera.setYaw(-90.0f);
            camera.setPitch(-15.0f); // Approx look down

            UMGEBUNG_LOG_INFO("Focused camera on entity {}", static_cast<uint32_t>(entity));
        }
    }

    void Application::processInput(float deltaTime) {
        GLFWwindow* nativeWindow = window_->getGLFWwindow();

        if (glfwGetKey(nativeWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(nativeWindow, true);
        }

        if (auto* viewport = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
            if (viewport->isFocused() && (glfwGetMouseButton(nativeWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)) {

                glfwSetInputMode(nativeWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

                if (glfwGetKey(nativeWindow, GLFW_KEY_W) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::FORWARD, deltaTime);
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_S) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::BACKWARD, deltaTime);
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_A) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::LEFT, deltaTime);
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_D) == GLFW_PRESS) {
                    getActiveCamera().processKeyboard(renderer::Camera_Movement::RIGHT, deltaTime); 
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