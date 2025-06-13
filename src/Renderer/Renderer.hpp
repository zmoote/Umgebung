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

        // Getters for ImGui
        VkInstance get_instance() const { return instance.instance; }
        VkPhysicalDevice get_physical_device() const { return physicalDevice; }
        VkDevice get_logical_device() const { return logicalDevice; }
        VkQueue get_graphics_queue() const { return graphicsQueue; }
        uint32_t get_graphics_queue_family() const { return graphicsQueueFamily; }
        VkRenderPass get_render_pass() const { return renderPass; }
        VkCommandBuffer get_current_command_buffer() const { return commandBuffers[currentImageIndex]; }

    private:
        void InitVulkan(Window* window);
        void CreateSwapchain();
        void CreateRenderPass();
        void CreateFramebuffers();
        void CreateCommandBuffers();
        void InitSyncObjects();
        void RecreateSwapchain();
        void CleanupSyncObjects();

        static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

        vkb::Instance instance;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        vkb::Device device;
        VkDevice logicalDevice = VK_NULL_HANDLE;
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        VkQueue presentQueue = VK_NULL_HANDLE;
        uint32_t graphicsQueueFamily = 0;
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        std::vector<VkImage> swapchainImages;
        std::vector<VkImageView> swapchainImageViews;
        VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        std::vector<VkFramebuffer> framebuffers;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> commandBuffers;
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        uint32_t currentFrameIndex = 0;
        uint32_t currentImageIndex = 0;
        Window* window = nullptr;
        bool framebufferResized = false;
        bool isRecreatingSwapchain = false;
    };

} // namespace Umgebung