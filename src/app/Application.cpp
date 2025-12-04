/**
 * @file Application.cpp
 * @brief Implements the Application class.
 */
#include "umgebung/app/Application.hpp"
#include "umgebung/renderer/Mesh.hpp"
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

        scene_ = std::make_unique<scene::Scene>();
        renderSystem_ = std::make_unique<ecs::systems::RenderSystem>(renderer_.get());
        assetSystem_ = std::make_unique<ecs::systems::AssetSystem>(renderer_->getModelLoader());

        physicsSystem_ = std::make_unique<ecs::systems::PhysicsSystem>();
        physicsSystem_->init(window_->getGLFWwindow());

        debugRenderer_ = std::make_unique<renderer::DebugRenderer>();
        debugRenderer_->init();
        debugRenderSystem_ = std::make_unique<ecs::systems::DebugRenderSystem>(debugRenderer_.get());

        framebuffer_ = std::make_unique<renderer::Framebuffer>(1280, 720);

        sceneSerializer_ = std::make_unique<scene::SceneSerializer>(scene_.get(), renderer_.get());
        sceneSerializer_->deserialize("assets/scenes/default.umgebung");

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

        // Spawn Test Micro-Particles (As Entities)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distPos(-10.0f, 10.0f);
        std::uniform_real_distribution<float> distHeight(5.0f, 20.0f);
        
        for (int i = 0; i < 1000; ++i) { // Reduced to 1000 for ECS/UI sanity check
            auto entity = scene_->createEntity();
            scene_->getRegistry().replace<ecs::components::Name>(entity, "Particle_" + std::to_string(i));
            auto& transform = scene_->getRegistry().get<ecs::components::Transform>(entity);
            transform.position = { distPos(rng), distHeight(rng), distPos(rng) };
            transform.scale = { 0.05f, 0.05f, 0.05f };
            
            scene_->getRegistry().emplace<ecs::components::MicroBody>(entity);
            scene_->getRegistry().emplace<ecs::components::ScaleComponent>(entity, ecs::components::ScaleType::Micro);
        }

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
            sceneSerializer_->serialize("assets/scenes/temp.umgebung");
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
        sceneSerializer_->deserialize("assets/scenes/temp.umgebung");
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

            if (auto* viewportPanel = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
                glm::vec2 viewportSize = viewportPanel->getSize();
                if (framebuffer_->getWidth() != viewportSize.x || framebuffer_->getHeight() != viewportSize.y) {
                    if (viewportSize.x > 0 && viewportSize.y > 0) {
                        framebuffer_->resize(viewportSize.x, viewportSize.y);
                        renderer_->getCamera().setPerspective(
                            glm::radians(45.0f),
                            viewportSize.x / viewportSize.y,
                            0.1f, 1000.0f
                        );
                    }
                }
            }

            framebuffer_->bind();
            window_->clear();
            assetSystem_->onUpdate(*scene_);
            
            if (state_ == AppState::Simulate) {
                physicsSystem_->update(scene_->getRegistry(), deltaTime_, renderer_->getCamera().getPosition());
            }
            
            renderSystem_->onUpdate(*scene_);
            
            debugRenderer_->beginFrame(renderer_->getCamera());
            debugRenderSystem_->onUpdate(scene_->getRegistry());

            // Draw Micro-Particles (as Entities)
            if (state_ == AppState::Simulate || state_ == AppState::Paused) {
                 auto view = scene_->getRegistry().view<ecs::components::MicroBody, ecs::components::Transform>();
                 std::vector<glm::vec3> particlePositions;
                 particlePositions.reserve(std::distance(view.begin(), view.end()));
                 for (auto entity : view) {
                     particlePositions.push_back(view.get<ecs::components::Transform>(entity).position);
                 }
                 debugRenderer_->drawPoints(particlePositions, {0.0f, 1.0f, 1.0f, 1.0f}); // Cyan particles
            }

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

    void Application::processInput(float deltaTime) {
        GLFWwindow* nativeWindow = window_->getGLFWwindow();

        if (glfwGetKey(nativeWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(nativeWindow, true);
        }

        if (auto* viewport = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
            if (viewport->isFocused() && (glfwGetMouseButton(nativeWindow, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)) {

                glfwSetInputMode(nativeWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

                if (glfwGetKey(nativeWindow, GLFW_KEY_W) == GLFW_PRESS) {
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::FORWARD, deltaTime);
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_S) == GLFW_PRESS) {
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::BACKWARD, deltaTime);
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_A) == GLFW_PRESS) {
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::LEFT, deltaTime);
                }
                    
                if (glfwGetKey(nativeWindow, GLFW_KEY_D) == GLFW_PRESS) {
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::RIGHT, deltaTime); 
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

                renderer_->getCamera().processMouseMovement(xoffset, yoffset);
            }
            else {
                firstMouse_ = true;
                glfwSetInputMode(nativeWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }
    }

}