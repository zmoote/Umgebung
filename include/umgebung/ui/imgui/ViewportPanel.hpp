#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include <glm/glm.hpp>

namespace Umgebung::ui::imgui {

    class ViewportPanel : public Panel {
    public:
        explicit ViewportPanel(renderer::Framebuffer* framebuffer);

        void onUIRender() override;

        glm::vec2 getSize() const { return size_; }

        bool isFocused() const { return focused_; }

    private:
        renderer::Framebuffer* framebuffer_ = nullptr;
        glm::vec2 size_{ 0.0f, 0.0f };

        bool focused_ = false;
    };

}