/**
 * @file ViewportPanel.hpp
 * @brief Contains the ViewportPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include <functional> // For std::function
#include <glm/glm.hpp>

namespace Umgebung::renderer { class Framebuffer; }
namespace Umgebung::app { enum class AppState; class Application; } // Forward declarations

namespace Umgebung::ui::imgui {

class ViewportPanel : public Panel {
public:
    ViewportPanel(renderer::Framebuffer* framebuffer, std::function<app::AppState()> getAppState);

    void onUIRender() override;
    
    glm::vec2 getSize() const { return size_; }

private:
    renderer::Framebuffer* framebuffer_ = nullptr;
    std::function<app::AppState()> getAppState_;
    glm::vec2 size_{ 0.0f, 0.0f };
};

} // namespace Umgebung::ui::imgui