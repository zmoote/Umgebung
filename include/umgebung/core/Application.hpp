#pragma once
#include <GLFW/glfw3.h>
#include <bgfx/bgfx.h>
#include <imgui.h>
#include <memory>
#include "scene/Scene.hpp"
#include "rendering/Renderer.hpp"
#include "ui/UI.hpp"

namespace umgebung::core {
    class Application {
    public:
        Application(int width, int height, const char* title);
        ~Application();

        void run();
        void set_view_mode(bool is_2d);

    private:
        void init_window();
        void init_bgfx();
        void init_imgui();
        void update(float delta_time);
        void render();

        GLFWwindow* window_;
        int width_, height_;
        bool is_2d_view_;
        std::unique_ptr<Scene> scene_;
        std::unique_ptr<Renderer> renderer_;
        std::unique_ptr<UI> ui_;
    };
}