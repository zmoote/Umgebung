// --- Renderer/Renderer.cpp ---
#include "Renderer.hpp"
#include "../Platform/Window.hpp"
#include "../Core/Logger.hpp"

namespace Umgebung {

    void Renderer::Init(Window* window) {
        Logger::GetCoreLogger()->info("Renderer initialized (stub).");
        // Vulkan setup will go here
    }

    void Renderer::BeginFrame() {
        // Begin Vulkan command buffer
    }

    void Renderer::EndFrame() {
        // Submit Vulkan command buffer
    }

    void Renderer::Cleanup() {
        Logger::GetCoreLogger()->info("Renderer cleanup (stub).");
    }

} // namespace Umgebung