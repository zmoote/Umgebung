#pragma once
#include "umgebung/ui/imgui/Panel.hpp" // FIX: Include the base class header

// Forward declarations are better here to reduce compile times
namespace umgebung {
    namespace renderer {
        class Framebuffer;
        class Camera;
    }
}

namespace umgebung::ui::imgui {
    class ViewportPanel : public Panel {
    public:
        ViewportPanel(renderer::Framebuffer& framebuffer, renderer::Camera& camera);
        void render() override;
    private:
        renderer::Framebuffer& m_framebuffer;
        renderer::Camera& m_camera;
    };
}