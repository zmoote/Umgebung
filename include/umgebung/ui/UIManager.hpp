#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <vector>
#include <memory>

// --- Forward-declare missing types ---
struct GLFWwindow;
namespace Umgebung::scene { class Scene; }
namespace Umgebung::renderer { class Framebuffer; }
namespace Umgebung::ui::imgui { class ViewportPanel; } // Forward-declare ViewportPanel

namespace Umgebung::ui {

    class UIManager {
    public:
        UIManager();
        ~UIManager();

        // Correct the init signature
        void init(GLFWwindow* window, scene::Scene* scene, renderer::Framebuffer* framebuffer);
        void shutdown();
        void beginFrame();
        void endFrame();

        // Add the missing getter
        imgui::ViewportPanel* getViewportPanel();

    private:
        void setupDockspace();
        scene::Scene* scene_ = nullptr;
        std::vector<std::unique_ptr<imgui::Panel>> panels_;
    };

} // namespace Umgebung::ui