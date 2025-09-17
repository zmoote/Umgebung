#include "umgebung/app/Application.hpp"

// Add includes for the new classes we need to use
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/Transform.hpp"

// Note: No need for <glad/glad.h> here anymore as the Window class handles it.

namespace Umgebung::app {

    Application::Application() {
        // Constructor is now empty. Initialization is handled by init().
    }

    // init() now returns an int to signal success/failure, matching Main.cpp
    int Application::init() {
        window_ = std::make_unique<ui::Window>(800, 600, "Umgebung");
        if (window_->init() != 0) {
            // UMGEBUNG_LOG_CRIT("Failed to initialize window!");
            return -1;
        }

        renderer_ = std::make_unique<renderer::Renderer>();
        renderer_->init();

        scene_ = std::make_unique<scene::Scene>();
        renderSystem_ = std::make_unique<ecs::systems::RenderSystem>(renderer_.get());

        createTriangleEntity();
        return 0;
    }

    void Application::run() {
        // This loop now correctly uses your Window class methods
        while (!window_->shouldClose()) {
            // Start the frame (polls for events)
            window_->beginFrame();

            // Clear the screen
            window_->clear();

            // --- Our Scene Rendering ---
            renderSystem_->onUpdate(*scene_);
            // -------------------------

            // In the future, your ImGui rendering calls will go here
            // window_->beginImGuiFrame();
            // ...
            // window_->endImGuiFrame();

            // End the frame (swaps buffers)
            window_->endFrame();
        }
    }

    void Application::shutdown() {
        // Clean up resources if necessary
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