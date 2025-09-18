#pragma once

#include "umgebung/ui/Window.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/systems/RenderSystem.hpp"

#include <memory>

namespace Umgebung::app {

    class Application {
    public:
        Application();
        void run();
        int init();
        void shutdown();

        // --- Add this new public function ---
        void onWindowResize(int width, int height);

    private:
        void createTriangleEntity(); // Helper to create our test object

        std::unique_ptr<ui::Window> window_;
        std::unique_ptr<renderer::Renderer> renderer_;

        // --- New ECS Members ---
        std::unique_ptr<scene::Scene> scene_;
        std::unique_ptr<ecs::systems::RenderSystem> renderSystem_;

        bool running_ = true;
    };

} // namespace Umgebung::app