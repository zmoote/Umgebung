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

    class UIManager {
    public:
        using AppCallbackFn = std::function<void()>;

        UIManager();
        ~UIManager();

        void init(GLFWwindow* window, scene::Scene* scene, renderer::Framebuffer* framebuffer, renderer::Renderer* renderer);
        void shutdown();
        void beginFrame();
        void endFrame();

        template<typename T>
        T* getPanel() {
            for (const auto& panel : panels_) {
                if (T* p = dynamic_cast<T*>(panel.get())) {
                    return p;
                }
            }
            return nullptr;
        }

        void setAppCallback(const AppCallbackFn& callback);

    private:
        void setupDockspace();
        scene::Scene* scene_ = nullptr;
        std::vector<std::unique_ptr<imgui::Panel>> panels_;

        bool firstFrame_ = true;

        AppCallbackFn appCallback_ = nullptr;

        std::unique_ptr<scene::SceneSerializer> m_SceneSerializer;

        renderer::Renderer* m_Renderer{ nullptr };
    };

}