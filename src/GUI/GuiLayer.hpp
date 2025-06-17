#pragma once
#include "../Core/Layer.hpp"
#include <vulkan/vulkan.h>

namespace Umgebung {

    class Renderer;
    class Window;

    class GuiLayer : public Layer {
    public:
        GuiLayer(Renderer* pRenderer, Window* pWindow); // Add constructor
        ~GuiLayer() override = default;

        void OnAttach() override;
        void OnDetach() override;
        void OnImGuiRender() override;

    private:
        Renderer* renderer;
        Window* window;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    };

} // namespace Umgebung