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
        
    };

    /**
     * @brief The main application class.
     * 
     * This class manages the main application loop, window, renderer, scene, and UI.
     */
    class Application {
    public:
        /**
         * @brief Construct a new Application object.
         */
        Application();

        /**
         * @brief Destroy the Application object.
         */
        ~Application();

        /**
         * @brief Runs the main application loop.
         */
        void run();

        /**
         * @brief Initializes the application.
         * @return 0 on success, -1 on failure.
         */
        int init();

        /**
         * @brief Closes the application.
         */
        void close();

    private:
        /**
         * @brief Shuts down the application and releases resources.
         */
        void shutdown();

        /**
         * @brief Creates a simple triangle entity for testing purposes.
         */
        void createTriangleEntity();

        std::unique_ptr<ui::Window> window_; ///< The application window.
        std::unique_ptr<renderer::Renderer> renderer_; ///< The renderer.

        std::unique_ptr<scene::Scene> scene_; ///< The scene.
        std::unique_ptr<ecs::systems::RenderSystem> renderSystem_; ///< The render system.

        std::unique_ptr<ui::UIManager> uiManager_; ///< The UI manager.

        std::unique_ptr<renderer::Framebuffer> framebuffer_; ///< The framebuffer.

        /**
         * @brief Processes user input.
         * @param deltaTime The time since the last frame.
         */
        void processInput(float deltaTime);

        float deltaTime_ = 0.0f; ///< The time since the last frame.
        float lastFrame_ = 0.0f; ///< The time of the last frame.

        bool firstMouse_ = true; ///< Whether this is the first mouse input.
        double lastX_ = 0.0; ///< The last mouse X position.
        double lastY_ = 0.0; ///< The last mouse Y position.
    };

}