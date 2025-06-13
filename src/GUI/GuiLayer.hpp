#pragma once
#include "../Core/Layer.hpp"
#include <vulkan/vulkan.h> // Added for VkDescriptorPool

namespace Umgebung {

    class Renderer;
    class Window;

    class GuiLayer : public Layer {
    public:
        void OnAttach(Renderer* renderer, Window* window);
        void OnDetach() override;
        void OnImGuiRender() override;

    private:
        Renderer* renderer = nullptr;
        Window* window = nullptr;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    };

} // namespace Umgebung