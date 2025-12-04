#pragma once

#include "umgebung/ui/Window.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/systems/RenderSystem.hpp"
#include "umgebung/ecs/systems/AssetSystem.hpp"
#include "umgebung/ecs/systems/PhysicsSystem.hpp"
#include "umgebung/ecs/systems/DebugRenderSystem.hpp"
#include "umgebung/ecs/systems/ObserverSystem.hpp"
#include "umgebung/renderer/DebugRenderer.hpp"
#include "umgebung/ui/UIManager.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include "umgebung/scene/SceneSerializer.hpp"

#include <memory>

namespace Umgebung::app {

    enum class AppState {
        Editor,
        Simulate,
        Paused
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

        void onPlay();
        void onStop();
        void onPause();

        AppState getState() const { return state_; }

        renderer::Camera& getActiveCamera();
        renderer::Camera& getEditorCamera() { return *editorCamera_; }

        /**
         * @brief Focuses the active camera on the specified entity.
         * @param entity The entity to focus on.
         */
        void focusOnEntity(entt::entity entity);

    private:
        /**
         * @brief Shuts down the application and releases resources.
         */
        void shutdown();

        std::unique_ptr<ui::Window> window_; ///< The application window.
        std::unique_ptr<renderer::Renderer> renderer_; ///< The renderer.

        std::unique_ptr<scene::Scene> scene_; ///< The scene.
        std::unique_ptr<scene::SceneSerializer> sceneSerializer_; ///< The scene serializer.
        std::unique_ptr<ecs::systems::RenderSystem> renderSystem_; ///< The render system.
        std::unique_ptr<ecs::systems::AssetSystem> assetSystem_; ///< The asset system.
        std::unique_ptr<ecs::systems::PhysicsSystem> physicsSystem_; ///< The physics system.
        std::unique_ptr<ecs::systems::ObserverSystem> observerSystem_; ///< The observer system.
        std::unique_ptr<ecs::systems::DebugRenderSystem> debugRenderSystem_; ///< The debug render system.
        std::unique_ptr<renderer::DebugRenderer> debugRenderer_; ///< The debug renderer.

        std::unique_ptr<ui::UIManager> uiManager_; ///< The UI manager.

        std::unique_ptr<renderer::Framebuffer> framebuffer_; ///< The framebuffer.

        std::unique_ptr<renderer::Camera> editorCamera_; ///< The editor camera.

        AppState state_ = AppState::Editor;

        /**
         * @brief Processes user input.
         * @param deltaTime The time since the last frame.
         */
        void processInput(float deltaTime);
        void updateWindowTitle();

        float deltaTime_ = 0.0f; ///< The time since the last frame.
        float lastFrame_ = 0.0f; ///< The time of the last frame.

        bool firstMouse_ = true; ///< Whether this is the first mouse input.
        double lastX_ = 0.0; ///< The last mouse X position.
        double lastY_ = 0.0; ///< The last mouse Y position.
    };

}