#pragma once
#include <bgfx/bgfx.h>
#include <glm/glm.hpp>
#include "scene/Scene.hpp"

namespace umgebung::rendering {
    class Renderer {
    public:
        Renderer();
        ~Renderer();
        void render(Scene& scene, bool is_2d_view);

    private:
        void setup_2d_view(const glm::mat4& view, const glm::mat4& proj);
        void setup_3d_view(const glm::mat4& view, const glm::mat4& proj);
        bgfx::ProgramHandle program_;
    };
}