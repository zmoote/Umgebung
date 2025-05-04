// --- Core/Application.cpp ---
#include "Application.hpp"
#include "../Platform/Window.hpp"
#include "../Renderer/Renderer.hpp"
#include "../GUI/GuiLayer.hpp"
#include "Logger.hpp"

namespace Umgebung {

    Application::Application() = default;
    Application::~Application() = default;

    void Application::Init() {
        Logger::GetCoreLogger()->info("Initializing Application...");
        window = std::make_unique<Window>();
        window->Create("Umgebung", 1280, 720);

        renderer = std::make_unique<Renderer>();
        renderer->Init(window.get());

        gui = std::make_unique<GuiLayer>();
        gui->OnAttach(renderer.get());
    }

    void Application::Run() {
        while (!window->ShouldClose()) {
            window->PollEvents();
            renderer->BeginFrame();

            for (auto* layer : layerStack) {
                layer->OnUpdate(0.016f); // placeholder timestep
                layer->OnRender();
            }

            gui->OnImGuiRender();
            renderer->EndFrame();
        }
    }

    void Application::Shutdown() {
        gui->OnDetach();
        renderer->Cleanup();
        Logger::GetCoreLogger()->info("Shutting down Application.");
    }

} // namespace Umgebung