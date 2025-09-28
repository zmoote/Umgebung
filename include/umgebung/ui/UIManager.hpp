#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <vector>
#include <memory>
#include <functional>

// --- Forward-declare missing types ---
struct GLFWwindow;
namespace Umgebung::scene { class Scene; }
namespace Umgebung::renderer { class Framebuffer; }
namespace Umgebung::ui::imgui { class ViewportPanel; } // Forward-declare ViewportPanel

namespace Umgebung::ui {

    class UIManager {
    public:
        // --- Add a new callback type for application events ---
        using AppCallbackFn = std::function<void()>;

        UIManager();
        ~UIManager();

        // Correct the init signature
        void init(GLFWwindow* window, scene::Scene* scene, renderer::Framebuffer* framebuffer);
        void shutdown();
        void beginFrame();
        void endFrame();

        template<typename T>
        T* getPanel() {
            for (const auto& panel : panels_) {
                if (T* p = dynamic_cast<T*>(panel.get())) {
                    return p; // Returns a pointer to the found panel
                }
            }
            return nullptr; // Return null if not found
        }

        // --- Add a setter for our new callback ---
        void setAppCallback(const AppCallbackFn& callback);

    private:
        void setupDockspace();
        scene::Scene* scene_ = nullptr;
        std::vector<std::unique_ptr<imgui::Panel>> panels_;

        // --- Add this flag ---
        bool firstFrame_ = true;

        // --- Add a member to store the callback ---
        AppCallbackFn appCallback_ = nullptr;
    };

} // namespace Umgebung::ui