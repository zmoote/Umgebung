#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include <glm/glm.hpp>

namespace Umgebung::ui::imgui {

    class ViewportPanel : public Panel {
    public:
        // Pass the name up to the base Panel constructor
        explicit ViewportPanel(renderer::Framebuffer* framebuffer);

        void onUIRender() override; // This will now correctly override the base method

        glm::vec2 getSize() const { return size_; }

    private:
        renderer::Framebuffer* framebuffer_ = nullptr;
        glm::vec2 size_{ 0.0f, 0.0f };
    };

} // namespace Umgebung::ui::imgui