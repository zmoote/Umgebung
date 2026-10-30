#pragma once

#include "umgebung/core/SystemProfiler.hpp"
#include <nvrhi/nvrhi.h>
#include <vulkan/vulkan_core.h>
#include <vector>

struct GLFWwindow;

namespace umgebung::graphics {

    class RenderDevice {
    public:
        RenderDevice();
        ~RenderDevice();

        bool Initialize(const core::HardwareCapabilities& caps);
        void Shutdown();

        bool BeginFrame();
        void Present();

        GLFWwindow* GetWindow() const { return m_window; }
        nvrhi::IDevice* GetNVRHIDevice() const { return m_nvrhiDevice; }

    private:
        bool CreateSwapchain();
        void DestroySwapchain();

        GLFWwindow* m_window{nullptr};

        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkSurfaceKHR m_vkSurface{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        VkQueue m_graphicsQueue{VK_NULL_HANDLE};
        uint32_t m_graphicsQueueFamily{0};

        VkSwapchainKHR m_swapChain{VK_NULL_HANDLE};
        VkFormat m_swapChainImageFormat{VK_FORMAT_B8G8R8A8_UNORM};
        VkExtent2D m_swapChainExtent{1280, 720};
        std::vector<VkImage> m_swapChainImages;
        std::vector<nvrhi::TextureHandle> m_swapChainTextures;
        uint32_t m_currentImageIndex{0};

        nvrhi::DeviceHandle m_nvrhiDevice;
    };

} // namespace umgebung::graphics