#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <vector>
#include <memory>
#include <functional>

struct GLFWwindow;
namespace Umgebung::scene { class Scene; class SceneSerializer; }
namespace Umgebung::renderer { class Framebuffer; class Renderer; }
namespace Umgebung::ui::imgui { class ViewportPanel; }

namespace Umgebung::ui {

    /**
 * @file UIManager.hpp
 * @brief Contains the UIManager class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <vector>
#include <memory>
#include <functional>

struct GLFWwindow;
namespace Umgebung::scene { class Scene; class SceneSerializer; }
namespace Umgebung::renderer { class Framebuffer; class Renderer; }
namespace Umgebung::ui::imgui { class ViewportPanel; }

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
         * @param scene The scene.
         * @param framebuffer The framebuffer.
         * @param renderer The renderer.
         */
        void init(GLFWwindow* window, scene::Scene* scene, renderer::Framebuffer* framebuffer, renderer::Renderer* renderer);

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

    private:
        /**
         * @brief Sets up the ImGui dockspace.
         */
        void setupDockspace();
        scene::Scene* scene_ = nullptr; ///< The scene.
        std::vector<std::unique_ptr<imgui::Panel>> panels_; ///< The UI panels.

        bool firstFrame_ = true; ///< Whether this is the first frame.

        AppCallbackFn appCallback_ = nullptr; ///< The application callback function.

        std::unique_ptr<scene::SceneSerializer> m_SceneSerializer; ///< The scene serializer.

        renderer::Renderer* m_Renderer{ nullptr }; ///< The renderer.
    };

}

}