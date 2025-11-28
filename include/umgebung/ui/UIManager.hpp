#pragma once
#include "umgebung/ui/imgui/FilePickerPanel.hpp"

#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/ecs/systems/DebugRenderSystem.hpp"
#include <vector>
#include <memory>
#include <filesystem>
#include <functional>

struct GLFWwindow;
namespace Umgebung::scene { class Scene; class SceneSerializer; }
namespace Umgebung::renderer { class Framebuffer; class Renderer; }
namespace Umgebung::ui::imgui { class ViewportPanel; }
namespace Umgebung::app { class Application; }

namespace Umgebung::ui {

    /**
     * @brief A class that manages the ImGui user interface.
     */
    class UIManager {
    public:
        using AppCallbackFn = std::function<void()>;

        /**
         * @brief Construct a new UIManager object.
         */
        UIManager();

        /**
         * @brief Destroy the UIManager object.
         */
        ~UIManager();

        /**
         * @brief Initializes the UI manager.
         * 
         * @param window The GLFW window.
         * @param app The application instance.
         * @param scene The scene.
         * @param framebuffer The framebuffer.
         * @param renderer The renderer.
         * @param debugRenderSystem The debug render system.
         * @param sceneSerializer The scene serializer.
         */
        void init(GLFWwindow* window, app::Application* app, scene::Scene* scene, renderer::Framebuffer* framebuffer, renderer::Renderer* renderer, ecs::systems::DebugRenderSystem* debugRenderSystem, scene::SceneSerializer* sceneSerializer);

        /**
         * @brief Shuts down the UI manager.
         */
        void shutdown();

        /**
         * @brief Begins a new ImGui frame.
         */
        void beginFrame();

        /**
         * @brief Ends the current ImGui frame.
         */
        void endFrame();

        void openFilePicker(const std::string& title, const std::string& buttonLabel, imgui::FilePickerPanel::FileSelectedCallback callback, const std::vector<std::string>& extensions, const std::filesystem::path& startPath = {});

        /**
         * @brief Get a panel of a specific type.
         * 
         * @tparam T The type of the panel.
         * @return T* A pointer to the panel, or nullptr if it doesn't exist.
         */
        template<typename T>
        T* getPanel() {
            for (const auto& panel : panels_) {
                if (T* p = dynamic_cast<T*>(panel.get())) {
                    return p;
                }
            }
            return nullptr;
        }

        /**
         * @brief Set the App Callback object.
         * 
         * @param callback The callback function.
         */
        void setAppCallback(const AppCallbackFn& callback);
        void setStateCallbacks(std::function<void()> onPlay, std::function<void()> onStop, std::function<void()> onPause);

    private:
        /**
         * @brief Sets up the ImGui dockspace.
         */
        void setupDockspace();
        scene::Scene* scene_ = nullptr; ///< The scene.
        std::vector<std::unique_ptr<imgui::Panel>> panels_; ///< The UI panels.

        bool firstFrame_ = true; ///< Whether this is the first frame.

        AppCallbackFn appCallback_ = nullptr; ///< The application callback function.
        std::function<void()> onPlayCallback_ = nullptr;
        std::function<void()> onStopCallback_ = nullptr;
        std::function<void()> onPauseCallback_ = nullptr;

        scene::SceneSerializer* m_SceneSerializer = nullptr; ///< The scene serializer.

        renderer::Renderer* m_Renderer{ nullptr }; ///< The renderer.
        ecs::systems::DebugRenderSystem* debugRenderSystem_{ nullptr }; ///< The debug render system.
        std::unique_ptr<imgui::FilePickerPanel> filePickerPanel_; ///< The file picker panel.
        std::filesystem::path currentScenePath_; ///< The path to the current scene file.

        app::Application* app_ = nullptr; ///< Pointer to the main application instance.
    };

}
