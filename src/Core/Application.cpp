#include "Application.hpp"
#include "../Core/Logger.hpp"
#include "../Platform/Window.hpp"
#include "../Renderer/Renderer.hpp"
#include "../GUI/GuiLayer.hpp"
#include <memory>

namespace Umgebung {

    Application::Application() = default;

    Application::~Application() = default;

    void Application::Init() {
        Logger::GetCoreLogger()->info("Initializing Application...");
        window = std::make_unique<Window>();
        window->Create("Umgebung", 1280, 720);

        renderer = std::make_unique<Renderer>();
        renderer->Init(window.get());

        // Push GuiLayer into LayerStack
        layerStack.PushLayer(new GuiLayer(renderer.get(), window.get()));
    }

    void Application::Run() {
        Logger::GetCoreLogger()->info("Starting main loop.");
        while (!window->ShouldClose()) {
            Logger::GetCoreLogger()->info("Polling events.");
            window->PollEvents();
            if (renderer->BeginFrame(window.get())) {
                Logger::GetCoreLogger()->info("Calling ImGui render.");
                for (Layer* layer : layerStack) {
                    layer->OnImGuiRender();
                }
                Logger::GetCoreLogger()->info("Calling EndFrame.");
                renderer->EndFrame();
            }
            else {
                Logger::GetCoreLogger()->info("BeginFrame returned false, skipping frame.");
            }
        }
        Logger::GetCoreLogger()->info("Main loop exited.");
    }

    void Application::Shutdown() {
        Logger::GetCoreLogger()->info("Shutting down application...");
        // Clean up LayerStack (calls OnDetach for each layer)
        while (!layerStack.empty()) {
            Layer* layer = layerStack.back();
            layerStack.PopLayer(layer);
            delete layer; // Delete the layer since it was created with new
        }
        renderer->Cleanup();
        Logger::GetCoreLogger()->info("Application shutdown.");
    }

} // namespace Umgebung