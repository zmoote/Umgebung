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
#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ecs/components/Name.hpp"

#include <GLFW/glfw3.h>

namespace Umgebung::app {

    Application::Application() = default;

    Application::~Application() {
        shutdown();
    }

    int Application::init() {
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

        uiManager_ = std::make_unique<ui::UIManager>();
        uiManager_->init(window_->getGLFWwindow(), scene_.get(), framebuffer_.get(), renderer_.get(), debugRenderSystem_.get());

        uiManager_->setAppCallback([this]() {
            this->close();
            });

        createPhysicsTestScene();
        return 0;
    }

    void Application::run() {
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
            physicsSystem_->update(scene_->getRegistry(), deltaTime_);
            renderSystem_->onUpdate(*scene_);
            
            debugRenderer_->beginFrame(renderer_->getCamera());
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

    void Application::createPhysicsTestScene() {
        // Create a dynamic cube
        {
            auto cubeEntity = scene_->createEntity();
            auto& name = scene_->getRegistry().get<ecs::components::Name>(cubeEntity);
            name.name = "Falling Cube";

            auto& transform = scene_->getRegistry().get<ecs::components::Transform>(cubeEntity);
            transform.position = glm::vec3(0.0f, 5.0f, 0.0f);

            scene_->getRegistry().emplace<ecs::components::Renderable>(
                cubeEntity,
                nullptr, // Mesh is loaded by AssetSystem via meshTag
                glm::vec4{ 0.8f, 0.2f, 0.3f, 1.0f },
                "assets/models/Cube.glb"
            );

            auto& rigidBody = scene_->getRegistry().emplace<ecs::components::RigidBody>(cubeEntity);
            rigidBody.type = ecs::components::RigidBody::BodyType::Dynamic;
            rigidBody.mass = 10.0f;

            auto& collider = scene_->getRegistry().emplace<ecs::components::Collider>(cubeEntity);
            collider.type = ecs::components::Collider::ColliderType::Box;
        }

        // Create a static ground plane
        {
            auto groundEntity = scene_->createEntity();
            auto& name = scene_->getRegistry().get<ecs::components::Name>(groundEntity);
            name.name = "Ground";

            auto& transform = scene_->getRegistry().get<ecs::components::Transform>(groundEntity);
            transform.position = glm::vec3(0.0f, -2.0f, 0.0f);
            transform.scale = glm::vec3(10.0f, 0.5f, 10.0f);

            scene_->getRegistry().emplace<ecs::components::Renderable>(
                groundEntity,
                nullptr, // Mesh is loaded by AssetSystem via meshTag
                glm::vec4{ 0.5f, 0.5f, 0.5f, 1.0f },
                "assets/models/Cube.glb"
            );

            auto& rigidBody = scene_->getRegistry().emplace<ecs::components::RigidBody>(groundEntity);
            rigidBody.type = ecs::components::RigidBody::BodyType::Static;

            auto& collider = scene_->getRegistry().emplace<ecs::components::Collider>(groundEntity);
            collider.type = ecs::components::Collider::ColliderType::Box;
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