#pragma once

#include "vk_types.h"

struct SDL_Window* _window{ nullptr };

namespace Umgebung {
	struct FrameData {

		VkCommandPool _commandPool;
		VkCommandBuffer _mainCommandBuffer;
	};

	constexpr unsigned int FRAME_OVERLAP = 2;

	class VulkanEngine {
		public:
			bool _isInitialized{ false };
			int _frameNumber{ 0 };
			bool stop_rendering{ false };
			VkExtent2D _windowExtent{ 1700 , 900 };
			VkInstance _instance;// Vulkan library handle
			VkDebugUtilsMessengerEXT _debug_messenger;// Vulkan debug output handle
			VkPhysicalDevice _chosenGPU;// GPU chosen as the default device
			VkDevice _device; // Vulkan device for commands
			VkSurfaceKHR _surface;// Vulkan window surface
			VkSwapchainKHR _swapchain;
			VkFormat _swapchainImageFormat;
			
			FrameData _frames[FRAME_OVERLAP];

			FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

			VkQueue _graphicsQueue;
			uint32_t _graphicsQueueFamily;

			std::vector<VkImage> _swapchainImages;
			std::vector<VkImageView> _swapchainImageViews;
			VkExtent2D _swapchainExtent;

			static VulkanEngine& Get();

			void init();
			void cleanup();
			void draw();
			void run();
			void init_vulkan();
			void init_swapchain();
			void init_commands();
			void init_sync_structures();
			void create_swapchain(uint32_t width, uint32_t height);
			void destroy_swapchain();
	};
}