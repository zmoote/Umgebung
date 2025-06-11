#pragma once
#include <memory>
#include "../Core/LayerStack.hpp"

namespace Umgebung {

    class Window;
    class Renderer;
    class GuiLayer;

    class Application {
    public:
        Application();
        ~Application();

        void Init();
        void Run();
        void Shutdown();

    private:
        std::unique_ptr<Window> window;
        std::unique_ptr<Renderer> renderer;
        std::unique_ptr<GuiLayer> gui;
        LayerStack layerStack;
    };

} // namespace Umgebung