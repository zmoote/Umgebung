#pragma once
#include "../Platform/Window.hpp"
#include "../Renderer/Renderer.hpp"
#include "../Core/LayerStack.hpp"
#include <memory>

namespace Umgebung {

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
        LayerStack layerStack; // Replace std::unique_ptr<GuiLayer> with LayerStack
    };

} // namespace Umgebung