#pragma once
#include "umgebung/ui/imgui/Panel.hpp"

// Forward-declarations to reduce compile times
namespace Umgebung {
    namespace renderer {
        class Framebuffer;
        class Camera;
    }
}

namespace Umgebung::ui::imgui {
    class ViewportPanel : public Panel {
    public:
        ViewportPanel(renderer::Framebuffer& framebuffer, renderer::Camera& camera);
        void render() override;
    private:
        renderer::Framebuffer& m_framebuffer;
        renderer::Camera& m_camera;
    };
}