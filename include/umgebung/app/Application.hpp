#pragma once

#include "umgebung/ui/Window.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/systems/RenderSystem.hpp"
#include "umgebung/ui/UIManager.hpp" // <-- Add this include
#include "umgebung/renderer/Framebuffer.hpp" // <-- Add this

#include <memory>

namespace Umgebung::app {

    class Application {
    public:
        Application();
        ~Application(); // <-- Add destructor declaration
        void run();
        int init();
        void close(); // Public method to signal the app to close

    private:
        void shutdown(); // <-- Make shutdown private
        void createTriangleEntity(); // Helper to create our test object

        std::unique_ptr<ui::Window> window_;
        std::unique_ptr<renderer::Renderer> renderer_;

        // --- New ECS Members ---
        std::unique_ptr<scene::Scene> scene_;
        std::unique_ptr<ecs::systems::RenderSystem> renderSystem_;

        // --- Add the UIManager ---
        std::unique_ptr<ui::UIManager> uiManager_;

        // --- Add the Framebuffer ---
        std::unique_ptr<renderer::Framebuffer> framebuffer_;

        void processInput(float deltaTime); // <-- New

        // --- Add these for delta time calculation ---
        float deltaTime_ = 0.0f;
        float lastFrame_ = 0.0f;

        // --- ADD these members for mouse polling ---
        bool firstMouse_ = true;
        double lastX_ = 0.0;
        double lastY_ = 0.0;
    };

} // namespace Umgebung::app