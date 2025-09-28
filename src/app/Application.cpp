#include "umgebung/app/Application.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ui/imgui/ViewportPanel.hpp" // <-- ADD THIS INCLUDE

#include <GLFW/glfw3.h> // <-- Add include for keyboard input

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

        // --- Add this block to connect the callback ---
        // This tells the UIManager to call our Application::close() method
        // when its app callback is triggered.
        uiManager_->setAppCallback([this]() {
            this->close();
            });

        createTriangleEntity();
        return 0;
    }

    // Replace your entire Application::run() method with this corrected version.
    void Application::run() {
        while (!window_->shouldClose()) {
            // --- 1. Start the frame and calculate delta time ---
            window_->beginFrame(); // This polls events, which are queued up

            float currentFrame = static_cast<float>(glfwGetTime());
            deltaTime_ = currentFrame - lastFrame_;
            lastFrame_ = currentFrame;

            // --- 2. Render all ImGui windows ---
            // This is the most important change. We render the UI first to determine
            // which panel is focused for the CURRENT frame.
            uiManager_->beginFrame();
            uiManager_->endFrame();

            // --- 3. Process keyboard and mouse input LAST ---
            // Now that the UI has been rendered, isFocused() will give us
            // the correct state for this frame, eliminating the lag.
            processInput(deltaTime_);

            // --- 4. Handle Scene Rendering ---
            // Resize framebuffer if viewport size has changed
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

            // Render the 3D scene to our framebuffer
            framebuffer_->bind();
            window_->clear(); // Clear the framebuffer
            renderSystem_->onUpdate(*scene_);
            framebuffer_->unbind();

            // --- 5. End the main window frame ---
            window_->endFrame(); // Swaps buffers
        }
    }

    void Application::close() {
        // This function signals the main loop in run() to terminate.
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

    // Replace your entire processInput function with this one:
    void Application::processInput(float deltaTime) {
        GLFWwindow* nativeWindow = window_->getGLFWwindow();

        if (glfwGetKey(nativeWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(nativeWindow, true);

        if (auto* viewport = uiManager_->getPanel<ui::imgui::ViewportPanel>()) {
            if (viewport->isFocused()) {
                // --- Keyboard Input ---
                if (glfwGetKey(nativeWindow, GLFW_KEY_W) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::FORWARD, deltaTime);
                if (glfwGetKey(nativeWindow, GLFW_KEY_S) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::BACKWARD, deltaTime);
                if (glfwGetKey(nativeWindow, GLFW_KEY_A) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::LEFT, deltaTime);
                if (glfwGetKey(nativeWindow, GLFW_KEY_D) == GLFW_PRESS)
                    renderer_->getCamera().processKeyboard(renderer::Camera_Movement::RIGHT, deltaTime);

                // --- Mouse Input ---
                double xpos, ypos;
                glfwGetCursorPos(nativeWindow, &xpos, &ypos);

                if (firstMouse_) {
                    lastX_ = xpos;
                    lastY_ = ypos;
                    firstMouse_ = false;
                }

                double xoffset = xpos - lastX_;
                double yoffset = lastY_ - ypos; // reversed since y-coordinates go from bottom to top

                lastX_ = xpos;
                lastY_ = ypos;

                renderer_->getCamera().processMouseMovement(xoffset, yoffset);
            }
            else {
                // If the viewport is not focused, we must reset the firstMouse flag
                // so we don't get a huge jump when it becomes focused again.
                firstMouse_ = true;
            }
        }
    }

} // namespace Umgebung::app