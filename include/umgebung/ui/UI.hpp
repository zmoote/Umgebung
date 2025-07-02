#pragma once
#include <imgui.h>
#include "core/Application.hpp"

namespace umgebung::ui {
    class UI {
    public:
        UI(Application& app);
        void render();

    private:
        Application& app_;
    };
}