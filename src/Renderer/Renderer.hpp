#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vector>

namespace Umgebung {

    class Window;

    class Renderer {
    public:
        /**
         * @brief Initialize the Vulkan renderer with the given window.
         * @param window Pointer to the window object.
         */
        void Init(Window* window);

        /**
         * @brief Begin a new frame for rendering.
         * @param window Pointer to the window object.
         * @return True if the frame can be rendered, false otherwise.
         */
        bool BeginFrame(Window* window);

        /**
         * @brief End the current frame and present it.
         */
        void EndFrame();

        /**
         * @brief Clean up all Vulkan resources.
         */
        void Cleanup();

        // Getters for ImGui
        VkInstance get_instance() const { return instance.instance; }
        VkPhysicalDevice get_physical_device() const { return physicalDevice; }
        VkDevice get_logical_device() const { return logicalDevice; }
        VkQueue get_graphics_queue() const { return graphicsQueue; }
        uint32_t get_graphics_queue_family() const { return graphicsQueueFamily; }
        VkRenderPass get_render_pass() const { return renderPass; }
        VkCommandBuffer get_current_command_buffer() const { return commandBuffers[currentImageIndex]; }

        // --- Resource tracking methods ---
        void TrackBuffer(VkBuffer buffer);
        void TrackImage(VkImage image);
        void TrackMemory(VkDeviceMemory memory);

        // --- Resource untracking methods ---
        void UntrackBuffer(VkBuffer buffer);
        void UntrackImage(VkImage image);
        void UntrackMemory(VkDeviceMemory memory);

    private:
        void InitVulkan(Window* window);
        void CreateSwapchain();
        void CreateRenderPass();
        void CreateFramebuffers();
        void CreateCommandBuffers();
        void InitSyncObjects(); // <-- make sure this is declared here
        void RecreateSwapchain();
        void CleanupSyncObjects();

        // --- New helper methods for refactoring ---
        bool IsWindowMinimized(Window* pWindow) const;
        void HandleWindowResize(Window* pWindow);
        bool WaitAndResetFence();
        bool AcquireNextSwapchainImage();
        bool BeginCommandBuffer(VkCommandBuffer commandBuffer);
        void BeginRenderPass(VkCommandBuffer commandBuffer);
        void SubmitFrame(VkCommandBuffer commandBuffer);
        void PresentFrame();

        // --- Constants ---
        static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
        static constexpr uint64_t FENCE_TIMEOUT = UINT64_MAX;

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

        // --- Synchronization objects now per swapchain image ---
        std::vector<VkSemaphore> imageAvailableSemaphores; // size: swapchainImages.size()
        std::vector<VkSemaphore> renderFinishedSemaphores; // size: swapchainImages.size()
        std::vector<VkFence> inFlightFences;               // size: MAX_FRAMES_IN_FLIGHT

        uint32_t currentFrameIndex = 0;
        uint32_t currentImageIndex = 0;
        Window* window = nullptr;
        bool framebufferResized = false;
        bool isRecreatingSwapchain = false;

        // --- Resource tracking containers ---
        std::vector<VkBuffer> buffers;
        std::vector<VkImage> images;
        std::vector<VkDeviceMemory> memories;
    };

} // namespace Umgebung