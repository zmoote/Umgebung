#pragma once

#include "umgebung/ui/Window.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/systems/RenderSystem.hpp"
#include "umgebung/ui/UIManager.hpp"
#include "umgebung/renderer/Framebuffer.hpp"

#include <memory>

namespace Umgebung::app {

    enum class AppState {
        Editor,
        Simulate
    };

    class Application {
    public:
        Application();
        ~Application();
        void run();
        int init();
        void close();

    private:
        void shutdown();
        void createTriangleEntity();

        std::unique_ptr<ui::Window> window_;
        std::unique_ptr<renderer::Renderer> renderer_;

        std::unique_ptr<scene::Scene> scene_;
        std::unique_ptr<ecs::systems::RenderSystem> renderSystem_;

        std::unique_ptr<ui::UIManager> uiManager_;

        std::unique_ptr<renderer::Framebuffer> framebuffer_;

        void processInput(float deltaTime);

        float deltaTime_ = 0.0f;
        float lastFrame_ = 0.0f;

        bool firstMouse_ = true;
        double lastX_ = 0.0;
        double lastY_ = 0.0;
    };

}