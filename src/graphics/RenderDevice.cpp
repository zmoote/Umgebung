#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "umgebung/graphics/RenderDevice.hpp"
#include <nvrhi/vulkan.h>
#include <spdlog/spdlog.h>
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <cstring>
#include <algorithm>

namespace umgebung::graphics {

// Static debug callback to route Vulkan validation messages directly to spdlog
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        spdlog::error("[Vulkan Validation Error]: {}", pCallbackData->pMessage);
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        spdlog::warn("[Vulkan Validation Warning]: {}", pCallbackData->pMessage);
    } else {
        spdlog::info("[Vulkan Validation Info]: {}", pCallbackData->pMessage);
    }
    return VK_FALSE;
}

// Helper to check if standard validation layers are supported by the driver instance
static bool CheckValidationLayerSupport(const std::vector<const char*>& validationLayers) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }
    return true;
}

RenderDevice::RenderDevice() = default;

RenderDevice::~RenderDevice() {
    Shutdown();
}

bool RenderDevice::Initialize(const core::HardwareCapabilities& caps) {
    spdlog::info("Initializing RenderDevice (GLFW + Vulkan + NVRHI + Swapchain)...");

    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Capture all internal GLFW errors and pipe them directly into spdlog
    glfwSetErrorCallback([](int error, const char* description) {
        spdlog::error("GLFW Error [{}]: {}", error, description);
    });

#if UMGEBUNG_PLATFORM_LINUX
        const char* displayEnv = std::getenv("DISPLAY");
        const char* waylandEnv = std::getenv("WAYLAND_DISPLAY");
        const char* xdgRuntime = std::getenv("XDG_RUNTIME_DIR");
        spdlog::info("Linux Environment Check -> DISPLAY: {}, WAYLAND_DISPLAY: {}, XDG_RUNTIME_DIR: {}",
            displayEnv ? displayEnv : "unset",
            waylandEnv ? waylandEnv : "unset",
            xdgRuntime ? xdgRuntime : "unset");

    // Force X11 backend to run through XWayland on DISPLAY :1 for robust NVIDIA Vulkan extension support
    if (glfwPlatformSupported(GLFW_PLATFORM_X11)) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        spdlog::info("Configured GLFW to use the X11/XWayland platform.");
    } else {
        spdlog::warn("X11 platform not supported by this GLFW build.");
    }
#endif

    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW.");
        return false;
    }

    if (!glfwVulkanSupported()) {
        spdlog::error("GLFW reports Vulkan is not supported on this system.");
        glfwTerminate();
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    m_window = glfwCreateWindow(1280, 720, "Umgebung - Spacetime Simulation Framework", nullptr, nullptr);
    if (!m_window) {
        spdlog::error("Failed to create GLFW window.");
        glfwTerminate();
        return false;
    }

    // 1. Create Vulkan Instance & Validation Layers Setup
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (!glfwExtensions || glfwExtensionCount == 0) {
        spdlog::error("GLFW failed to return required Vulkan instance extensions. Is a display server running?");
        return false;
    }

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    for (const auto& ext : extensions) {
        spdlog::info("Requested Vulkan Instance Extension: {}", ext);
    }

    // Enable validation layers for debugging
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    bool enableValidation = true;
    if (enableValidation && !CheckValidationLayerSupport(validationLayers)) {
        spdlog::warn("Validation layers requested, but not available on this system. Disabling.");
        enableValidation = false;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Umgebung Simulation Engine";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "Umgebung";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidation) {
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidation) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        createInfo.pNext = &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    VkResult vkRes = vkCreateInstance(&createInfo, nullptr, &m_vkInstance);
    if (vkRes != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan instance. VkResult: {}", static_cast<int>(vkRes));
        return false;
    }

    spdlog::info("Checkpoint: Vulkan instance created successfully.");

    // Initialize Vulkan-Hpp instance dispatch pointers
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Instance(m_vkInstance));

    // 2. Create Window Surface
    vkRes = glfwCreateWindowSurface(m_vkInstance, m_window, nullptr, &m_vkSurface);
    if (vkRes != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan window surface. VkResult: {}", static_cast<int>(vkRes));
        return false;
    }
    spdlog::info("Checkpoint: Vulkan window surface created successfully.");

    // 3. Select Physical Device (Matching CUDA device UUID)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        spdlog::error("Failed to find GPUs with Vulkan support.");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, devices.data());

    for (auto dev : devices) {
        VkPhysicalDeviceIDProperties idProps{};
        idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

        VkPhysicalDeviceProperties2 deviceProps2{};
        deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProps2.pNext = &idProps;

        vkGetPhysicalDeviceProperties2(dev, &deviceProps2);

        if (std::memcmp(idProps.deviceUUID, caps.gpuUUID.bytes, VK_UUID_SIZE) == 0) {
            m_vkPhysicalDevice = dev;
            spdlog::info("Matched Vulkan physical device to CUDA device: {}", deviceProps2.properties.deviceName);
            break;
        }
    }

    if (m_vkPhysicalDevice == VK_NULL_HANDLE) {
        spdlog::warn("Could not match Vulkan device UUID precisely. Falling back to first available physical device.");
        m_vkPhysicalDevice = devices[0];
    }

    spdlog::info("Checkpoint: Physical device selected.");

    // 4. Find Queue Families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    int foundGraphicsQueue = -1;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(m_vkPhysicalDevice, i, m_vkSurface, &presentSupport);
            if (presentSupport) {
                foundGraphicsQueue = i;
                break;
            }
        }
    }

    if (foundGraphicsQueue == -1) {
        spdlog::error("Failed to find a suitable queue family with graphics and present support.");
        return false;
    }
    m_graphicsQueueFamily = static_cast<uint32_t>(foundGraphicsQueue);
    spdlog::info("Checkpoint: Graphics queue family found (index {}).", m_graphicsQueueFamily);

    // 5. Create Logical Device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(m_vkPhysicalDevice, &deviceCreateInfo, nullptr, &m_vkDevice) != VK_SUCCESS) {
        spdlog::error("Failed to create logical Vulkan device.");
        return false;
    }
    spdlog::info("Checkpoint: Logical Vulkan device created.");

    // Initialize Vulkan-Hpp device dispatch pointers
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Device(m_vkDevice));

    vkGetDeviceQueue(m_vkDevice, m_graphicsQueueFamily, 0, &m_graphicsQueue);

    // 6. Initialize NVRHI Device
    nvrhi::vulkan::DeviceDesc params{};
    params.instance = m_vkInstance;
    params.physicalDevice = m_vkPhysicalDevice;
    params.device = m_vkDevice;
    params.graphicsQueue = m_graphicsQueue;

    m_nvrhiDevice = nvrhi::vulkan::createDevice(params);
    if (!m_nvrhiDevice) {
        spdlog::error("Failed to create NVRHI device.");
        return false;
    }
    spdlog::info("Checkpoint: NVRHI device initialized successfully.");

    // 7. Create Swapchain and Textures
    if (!CreateSwapchain()) {
        spdlog::error("Failed to create swapchain.");
        return false;
    }
    spdlog::info("Checkpoint: Swapchain created successfully.");

    spdlog::info("RenderDevice successfully initialized with Swapchain.");
    return true;
}

bool RenderDevice::CreateSwapchain() {
    VkSurfaceCapabilitiesKHR surfCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkPhysicalDevice, m_vkSurface, &surfCaps);

    uint32_t imageCount = std::clamp(surfCaps.minImageCount + 1, surfCaps.minImageCount, surfCaps.maxImageCount > 0 ? surfCaps.maxImageCount : 8);

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_vkSurface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = m_swapChainImageFormat;
    createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    createInfo.imageExtent = m_swapChainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = surfCaps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(m_vkDevice, &createInfo, nullptr, &m_swapChain) != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan swapchain.");
        return false;
    }

    vkGetSwapchainImagesKHR(m_vkDevice, m_swapChain, &imageCount, nullptr);
    m_swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_vkDevice, m_swapChain, &imageCount, m_swapChainImages.data());

    m_swapChainTextures.resize(imageCount);
    for (size_t i = 0; i < imageCount; ++i) {
        nvrhi::TextureDesc texDesc;
        texDesc.dimension = nvrhi::TextureDimension::Texture2D;
        texDesc.width = m_swapChainExtent.width;
        texDesc.height = m_swapChainExtent.height;
        texDesc.format = nvrhi::Format::BGRA8_UNORM;
        texDesc.initialState = nvrhi::ResourceStates::Present;
        texDesc.keepInitialState = true;
        texDesc.debugName = "Swapchain Image";

        m_swapChainTextures[i] = m_nvrhiDevice->createHandleForNativeTexture(
            nvrhi::ObjectTypes::VK_Image,
            reinterpret_cast<void*>(m_swapChainImages[i]),
            texDesc
        );
    }

    return true;
}

void RenderDevice::DestroySwapchain() {
    m_swapChainTextures.clear();
    if (m_swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_vkDevice, m_swapChain, nullptr);
        m_swapChain = VK_NULL_HANDLE;
    }
}

void RenderDevice::Shutdown() {
    DestroySwapchain();
    if (m_nvrhiDevice) {
        m_nvrhiDevice = nullptr;
    }
    if (m_vkDevice) {
        vkDestroyDevice(m_vkDevice, nullptr);
        m_vkDevice = VK_NULL_HANDLE;
    }
    if (m_vkSurface) {
        vkDestroySurfaceKHR(m_vkInstance, m_vkSurface, nullptr);
        m_vkSurface = VK_NULL_HANDLE;
    }
    if (m_vkInstance) {
        vkDestroyInstance(m_vkInstance, nullptr);
        m_vkInstance = VK_NULL_HANDLE;
    }
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
    spdlog::info("RenderDevice shut down.");
}

} // namespace umgebung::graphics