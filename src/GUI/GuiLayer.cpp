// --- GUI/GuiLayer.cpp ---
#include "GuiLayer.hpp"
#include "../Renderer/Renderer.hpp"
#include "../Core/Logger.hpp"

namespace Umgebung {

    void GuiLayer::OnAttach(Renderer* renderer) {
        Logger::GetCoreLogger()->info("ImGui Layer attached (stub).");
        // Setup ImGui context and Vulkan bindings
    }

    void GuiLayer::OnDetach() {
        Logger::GetCoreLogger()->info("ImGui Layer detached (stub).");
        // Cleanup ImGui resources
    }

    void GuiLayer::OnImGuiRender() {
        // ImGui::Begin("Example"); ImGui::Text("Hello, Umgebung!"); ImGui::End();
    }

} // namespace Umgebung