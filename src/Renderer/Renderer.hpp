#pragma once

#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vector>

namespace Umgebung {

    class Window;

    class Renderer {
    public:
        void Init(Window* window);
        bool BeginFrame(Window* window);
        void EndFrame();
        void Cleanup();

    private:
        void InitVulkan(Window* window);
        void CreateSwapchain();
        void CreateRenderPass();
        void CreateFramebuffers();
        void CreateCommandBuffers();
        void InitSyncObjects();
        void RecreateSwapchain();
        void CleanupSyncObjects();

        // Vulkan Bootstrap objects
        vkb::InstanceBuilder builder;
        vkb::Instance bootstrapInstance;
        vkb::Device bootstrapDevice;

        // Vulkan core objects
        vkb::Instance instance;
        VkDebugUtilsMessengerEXT debugMessenger;
        VkPhysicalDevice physicalDevice;
        vkb::Device device;
        VkDevice logicalDevice;

        VkSurfaceKHR surface;
        VkSwapchainKHR swapchain;

        VkQueue graphicsQueue;
        VkQueue presentQueue;
        uint32_t graphicsQueueFamily;

        VkRenderPass renderPass;
        std::vector<VkFramebuffer> framebuffers;
        VkCommandPool commandPool;
        std::vector<VkCommandBuffer> commandBuffers;

        VkFormat swapchainImageFormat{};
        VkExtent2D swapchainExtent{};
        std::vector<VkImage> swapchainImages;
        std::vector<VkImageView> swapchainImageViews;

        uint32_t currentFrameIndex = 0;
        uint32_t currentImageIndex = 0;

        // Sync objects
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    };

} // namespace Umgebung