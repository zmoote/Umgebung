#include "umgebung/app/Application.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ui/imgui/ViewportPanel.hpp" // <-- ADD THIS INCLUDE

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

        createTriangleEntity();
        return 0;
    }

    void Application::run() {
        while (!window_->shouldClose()) {
            window_->beginFrame();

            // Resize framebuffer and camera if viewport size has changed
            if (auto* viewportPanel = uiManager_->getViewportPanel()) {
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

            // Render the UI
            uiManager_->beginFrame();
            uiManager_->endFrame();

            window_->endFrame();
        }
    }

    void Application::shutdown() {
        if (uiManager_) {
            uiManager_->shutdown();
        }
    }

    void Application::onWindowResize(int width, int height) {
        // This function is no longer needed as the viewport panel drives resizing.
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

} // namespace Umgebung::app