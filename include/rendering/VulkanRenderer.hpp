#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>

namespace Umgebung {
	
	class VulkanRenderer
	{
		public:
			VulkanRenderer();
			~VulkanRenderer();
	};
}