#include "umgebung/app/Application.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ui/imgui/ViewportPanel.hpp"

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

        framebuffer_ = std::make_unique<renderer::Framebuffer>(1280, 720);

        uiManager_ = std::make_unique<ui::UIManager>();
        uiManager_->init(window_->getGLFWwindow(), scene_.get(), framebuffer_.get());

        uiManager_->setAppCallback([this]() {
            this->close();
            });

        createTriangleEntity();
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
            renderSystem_->onUpdate(*scene_);
            framebuffer_->unbind();

            window_->endFrame();
        }
    }

    void Application::close() {
        glfwSetWindowShouldClose(window_->getGLFWwindow(), true);
    }

    void Application::shutdown() {
        if (uiManager_) {
            uiManager_->shutdown();
        }
    }

    void Application::createTriangleEntity() {
        std::vector<renderer::Vertex> vertices = {
            {{-0.5f, -0.5f, 0.0f}, {0,0,1}, {0,0}},
            {{ 0.5f, -0.5f, 0.0f}, {0,0,1}, {0,0}},
            {{ 0.0f,  0.5f, 0.0f}, {0,0,1}, {0,0}}
        };
        std::vector<uint32_t> indices = { 0, 1, 2 };

        auto triangleMesh = renderer::Mesh::create(vertices, indices);
        auto triangleEntity = scene_->createEntity();

        scene_->getRegistry().emplace<ecs::components::RenderableComponent>(
            triangleEntity,
            triangleMesh,
            glm::vec4{ 1.0f, 0.5f, 0.2f, 1.0f }
        );
    }

    void Application::processInput(float deltaTime) {
        GLFWwindow* nativeWindow = window_->getGLFWwindow();

        if (glfwGetKey(nativeWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(nativeWindow, true);

        if (auto* viewport = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
            if (viewport->isFocused()) {

                glfwSetInputMode(nativeWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

                if (glfwGetKey(nativeWindow, GLFW_KEY_W) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::FORWARD, deltaTime);
                if (glfwGetKey(nativeWindow, GLFW_KEY_S) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::BACKWARD, deltaTime);
                if (glfwGetKey(nativeWindow, GLFW_KEY_A) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::LEFT, deltaTime);
                if (glfwGetKey(nativeWindow, GLFW_KEY_D) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::RIGHT, deltaTime);

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
            }
        }
    }

}