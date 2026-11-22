/**
 * @file RenderDevice.cpp
 * @brief Implementation of the GUI-exclusive Vulkan 1.2 + NVIDIA NVRHI + Dear
 *        ImGui rendering device (`umgebung::graphics::RenderDevice`).
 *
 * @details
 * Responsibilities implemented in this translation unit:
 *   1. Defines and initializes the dynamic Vulkan-Hpp dispatch table required
 *      by statically linked `nvrhi_vk` (`VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE`).
 *   2. Creates a Vulkan 1.2 instance with GLFW WSI extensions and optional
 *      Khronos validation layers routed directly into `spdlog`.
 *   3. Matches the Vulkan physical GPU to the exact 16-byte hardware UUID of
 *      the primary CUDA GPU selected by `SystemProfiler`, ensuring rendering
 *      and compute share the same NVIDIA GPU on multi-GPU workstations.
 *   4. Enables Vulkan 1.2 timeline semaphores (`timelineSemaphore = VK_TRUE`)
 *      required by NVRHI's Vulkan command queue synchronization.
 *   5. Creates a Wayland-safe and X11/Win32-safe presentation swapchain
 *      (properly handling `currentExtent == UINT32_MAX` on Wayland), wraps
 *      swapchain images into `nvrhi::TextureHandle`s, builds a Vulkan render
 *      pass + framebuffers, and drives Dear ImGui 1.93+ (docking enabled).
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

// Enable and instantiate Vulkan-Hpp's dynamic dispatcher storage used by `nvrhi_vk`.
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "umgebung/graphics/RenderDevice.hpp"
#include "umgebung/graphics/Window.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <nvrhi/vulkan.h>
#include <spdlog/spdlog.h>

namespace umgebung::graphics {

    namespace {

        /** @brief Standard Khronos Vulkan validation layer name. */
        constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

        /**
         * @brief Vulkan debug messenger callback forwarding validation errors,
         *        warnings, and performance notes into `spdlog`.
         */
        VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
            const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
            void* /*userData*/
        ) {
            const char* message = (callbackData && callbackData->pMessage)
                ? callbackData->pMessage
                : "(empty Vulkan validation message)";

            if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
                spdlog::error("[Vulkan Validation] {}", message);
            } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
                spdlog::warn("[Vulkan Validation] {}", message);
            } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0) {
                spdlog::info("[Vulkan Validation] {}", message);
            } else {
                spdlog::debug("[Vulkan Validation] {}", message);
            }
            return VK_FALSE;
        }

        /**
         * @brief Dear ImGui Vulkan backend error callback logging non-success `VkResult`s.
         */
        void CheckImGuiVkResult(VkResult err) {
            if (err != VK_SUCCESS) {
                spdlog::error(
                    "[ImGui Vulkan] VkResult error: {} ({})",
                    nvrhi::vulkan::resultToString(err),
                    static_cast<int>(err)
                );
            }
        }

        /**
         * @brief Checks whether a named Vulkan instance layer is installed on the host.
         *
         * @param layerName Exact layer identifier (e.g., `"VK_LAYER_KHRONOS_validation"`).
         * @return `true` if available; `false` otherwise.
         */
        [[nodiscard]] bool IsInstanceLayerAvailable(const char* layerName) {
            std::uint32_t layerCount = 0U;
            if (vkEnumerateInstanceLayerProperties(&layerCount, nullptr) != VK_SUCCESS || layerCount == 0U) {
                return false;
            }
            std::vector<VkLayerProperties> available(layerCount);
            if (vkEnumerateInstanceLayerProperties(&layerCount, available.data()) != VK_SUCCESS) {
                return false;
            }
            for (const auto& layer : available) {
                if (std::strcmp(layer.layerName, layerName) == 0) {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Checks whether a named Vulkan instance extension is supported by the loader.
         */
        [[nodiscard]] bool IsInstanceExtensionAvailable(const char* extensionName) {
            std::uint32_t extCount = 0U;
            if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr) != VK_SUCCESS || extCount == 0U) {
                return false;
            }
            std::vector<VkExtensionProperties> available(extCount);
            if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, available.data()) != VK_SUCCESS) {
                return false;
            }
            for (const auto& ext : available) {
                if (std::strcmp(ext.extensionName, extensionName) == 0) {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Finds a queue family on `physicalDevice` supporting both
         *        graphics commands (`VK_QUEUE_GRAPHICS_BIT`) and presentation
         *        onto `surface`.
         *
         * @return Queue family index if supported; `std::nullopt` otherwise.
         */
        [[nodiscard]] std::optional<std::uint32_t> FindGraphicsPresentQueueFamily(
            VkPhysicalDevice physicalDevice,
            VkSurfaceKHR surface
        ) {
            std::uint32_t queueFamilyCount = 0U;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
            if (queueFamilyCount == 0U) {
                return std::nullopt;
            }

            std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, families.data());

            for (std::uint32_t i = 0U; i < queueFamilyCount; ++i) {
                const bool supportsGraphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U;
                VkBool32 supportsPresent = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &supportsPresent);
                if (supportsGraphics && supportsPresent == VK_TRUE) {
                    return i;
                }
            }
            return std::nullopt;
        }

        /**
         * @brief Maps a Vulkan swapchain `VkFormat` to the matching `nvrhi::Format`.
         */
        [[nodiscard]] nvrhi::Format VkFormatToNvrhiFormat(VkFormat format) noexcept {
            switch (format) {
                case VK_FORMAT_B8G8R8A8_UNORM:
                    return nvrhi::Format::BGRA8_UNORM;
                case VK_FORMAT_B8G8R8A8_SRGB:
                    return nvrhi::Format::SBGRA8_UNORM;
                case VK_FORMAT_R8G8B8A8_UNORM:
                    return nvrhi::Format::RGBA8_UNORM;
                case VK_FORMAT_R8G8B8A8_SRGB:
                    return nvrhi::Format::SRGBA8_UNORM;
                default:
                    return nvrhi::Format::BGRA8_UNORM;
            }
        }

        /**
         * @brief Applies a clean, dark astrophysics/engineering Dear ImGui color theme.
         */
        void ApplyUmgebungImGuiTheme() {
            ImGui::StyleColorsDark();
            ImGuiStyle& style = ImGui::GetStyle();
            style.WindowRounding = 5.0f;
            style.FrameRounding = 3.0f;
            style.GrabRounding = 3.0f;
            style.TabRounding = 4.0f;
            style.WindowBorderSize = 1.0f;
            style.FrameBorderSize = 0.0f;
            style.WindowPadding = ImVec2(10.0f, 10.0f);
            style.ItemSpacing = ImVec2(8.0f, 6.0f);

            ImVec4* colors = style.Colors;
            colors[ImGuiCol_WindowBg] = ImVec4(0.067f, 0.082f, 0.125f, 0.96f);
            colors[ImGuiCol_MenuBarBg] = ImVec4(0.052f, 0.065f, 0.102f, 1.00f);
            colors[ImGuiCol_TitleBg] = ImVec4(0.055f, 0.070f, 0.110f, 1.00f);
            colors[ImGuiCol_TitleBgActive] = ImVec4(0.098f, 0.145f, 0.240f, 1.00f);
            colors[ImGuiCol_Tab] = ImVec4(0.078f, 0.105f, 0.165f, 0.95f);
            colors[ImGuiCol_TabHovered] = ImVec4(0.185f, 0.300f, 0.490f, 1.00f);
            colors[ImGuiCol_TabSelected] = ImVec4(0.135f, 0.225f, 0.375f, 1.00f);
            colors[ImGuiCol_TabDimmed] = ImVec4(0.060f, 0.078f, 0.122f, 0.95f);
            colors[ImGuiCol_TabDimmedSelected] = ImVec4(0.098f, 0.150f, 0.245f, 1.00f);
            colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.043f, 0.055f, 0.090f, 1.00f);
            colors[ImGuiCol_DockingPreview] = ImVec4(0.240f, 0.560f, 0.920f, 0.65f);
            colors[ImGuiCol_Header] = ImVec4(0.145f, 0.245f, 0.400f, 0.70f);
            colors[ImGuiCol_HeaderHovered] = ImVec4(0.200f, 0.340f, 0.560f, 0.85f);
            colors[ImGuiCol_HeaderActive] = ImVec4(0.250f, 0.420f, 0.680f, 1.00f);
            colors[ImGuiCol_FrameBg] = ImVec4(0.105f, 0.130f, 0.195f, 0.90f);
            colors[ImGuiCol_PlotHistogram] = ImVec4(0.220f, 0.580f, 0.920f, 1.00f);
        }

    } // anonymous namespace

    /**
     * @class RenderDevice::NvrhiMessageBridge
     * @brief Concrete `nvrhi::IMessageCallback` forwarding NVRHI diagnostics to `spdlog`.
     */
    class RenderDevice::NvrhiMessageBridge final : public nvrhi::IMessageCallback {
    public:
        void message(nvrhi::MessageSeverity severity, const char* messageText) override {
            const char* text = messageText ? messageText : "(empty NVRHI message)";
            switch (severity) {
                case nvrhi::MessageSeverity::Info:
                    spdlog::info("[NVRHI] {}", text);
                    break;
                case nvrhi::MessageSeverity::Warning:
                    spdlog::warn("[NVRHI] {}", text);
                    break;
                case nvrhi::MessageSeverity::Error:
                    spdlog::error("[NVRHI] {}", text);
                    break;
                case nvrhi::MessageSeverity::Fatal:
                    spdlog::critical("[NVRHI FATAL] {}", text);
                    break;
            }
        }
    };

    RenderDevice::RenderDevice() = default;

    RenderDevice::~RenderDevice() {
        Shutdown();
    }

    bool RenderDevice::Initialize(
        Window& window,
        const core::HardwareCapabilities& caps,
        const core::HardwareAdaptationPlan& plan,
        const RenderDeviceConfig& config
    ) {
        if (m_initialized) {
            spdlog::warn("RenderDevice::Initialize called on an already initialized RenderDevice.");
            return false;
        }

        if (!window.IsInitialized()) {
            spdlog::error("RenderDevice::Initialize requires an already-initialized Window.");
            return false;
        }

        m_window = &window;
        m_clearColor = config.clearColor;
        m_swapchainNeedsRecreate = false;
        m_frameActive = false;

        spdlog::info("Initializing RenderDevice (Vulkan 1.4/1.3/1.2 + NVRHI + Swapchain + Dear ImGui)...");

        // Step 1: Bootstrap Vulkan-Hpp's global loader function pointers (`vkGetInstanceProcAddr`).
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        // Step 2..9: Execute initialization stages with deterministic rollback on any error.
        if (!CreateVulkanInstance(config) ||
            !CreateSurface() ||
            !SelectPhysicalDevice(caps, plan) ||
            !CreateLogicalDevice() ||
            !CreateNvrhiDevice() ||
            !CreateSwapchainResources() ||
            !CreateFrameSyncObjects() ||
            !InitializeImGui()) {
            spdlog::error("RenderDevice initialization failed; rolling back partial resources.");
            Shutdown();
            return false;
        }

        m_initialized = true;
        spdlog::info(
            "RenderDevice initialized cleanly (GPU=\"{}\", swapchain={}x{}, images={}, CUDA-UUID match={}).",
            m_selectedPhysicalDeviceName,
            m_swapchainExtent.width,
            m_swapchainExtent.height,
            m_swapchainImages.size(),
            m_matchedCudaUuid ? "YES" : "NO"
        );
        return true;
    }

    bool RenderDevice::CreateVulkanInstance(const RenderDeviceConfig& config) {
        std::vector<const char*> extensions = m_window->GetRequiredVulkanExtensions();
        if (extensions.empty()) {
            spdlog::error("GLFW returned zero required Vulkan WSI instance extensions.");
            return false;
        }

        // Enable Khronos validation layers if requested AND both layer and debug utils exist.
        m_validationEnabled = false;
        std::vector<const char*> layers;
        if (config.enableValidationLayers) {
            if (IsInstanceLayerAvailable(kValidationLayerName) &&
                IsInstanceExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                layers.push_back(kValidationLayerName);
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                m_validationEnabled = true;
            } else {
                spdlog::warn(
                    "Vulkan validation layer '{}' or '{}' not installed; continuing without validation.",
                    kValidationLayerName,
                    VK_EXT_DEBUG_UTILS_EXTENSION_NAME
                );
            }
        }

        m_enabledInstanceExtensions = extensions;
        for (const char* ext : m_enabledInstanceExtensions) {
            spdlog::debug("  Vulkan Instance Extension: {}", ext);
        }

        // Negotiate the highest Vulkan API version supported by both headers and
        // host loader (targeting Vulkan 1.4, with automatic fallback down to 1.2
        // for NVRHI's timeline semaphore baseline).
#if defined(VK_API_VERSION_1_4)
        constexpr std::uint32_t kTargetVulkanApiVersion = VK_API_VERSION_1_4;
#elif defined(VK_API_VERSION_1_3)
        constexpr std::uint32_t kTargetVulkanApiVersion = VK_API_VERSION_1_3;
#else
        constexpr std::uint32_t kTargetVulkanApiVersion = VK_API_VERSION_1_2;
#endif

        std::uint32_t loaderVersion = VK_API_VERSION_1_0;
        if (auto enumVersionFn = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
                vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion")
            )) {
            if (enumVersionFn(&loaderVersion) != VK_SUCCESS) {
                loaderVersion = VK_API_VERSION_1_0;
            }
        }

        const std::uint32_t loaderMajorMinor = VK_MAKE_API_VERSION(
            0U,
            VK_API_VERSION_MAJOR(loaderVersion),
            VK_API_VERSION_MINOR(loaderVersion),
            0U
        );
        m_instanceApiVersion = std::min(kTargetVulkanApiVersion, loaderMajorMinor);
        m_effectiveApiVersion = m_instanceApiVersion;

        if (m_instanceApiVersion < VK_API_VERSION_1_2) {
            spdlog::error(
                "Host Vulkan loader supports only {}.{}.{}; Vulkan >= 1.2 is required by NVRHI.",
                VK_API_VERSION_MAJOR(loaderVersion),
                VK_API_VERSION_MINOR(loaderVersion),
                VK_API_VERSION_PATCH(loaderVersion)
            );
            return false;
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Umgebung Reality Simulation Engine";
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.pEngineName = "Umgebung";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
        appInfo.apiVersion = m_instanceApiVersion;

        VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = VulkanDebugCallback;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(m_enabledInstanceExtensions.size());
        createInfo.ppEnabledExtensionNames = m_enabledInstanceExtensions.data();
        createInfo.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
        createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        createInfo.pNext = m_validationEnabled ? &debugInfo : nullptr;

        const VkResult res = vkCreateInstance(&createInfo, nullptr, &m_vkInstance);
        if (res != VK_SUCCESS) {
            spdlog::error("vkCreateInstance failed: {} ({}).", nvrhi::vulkan::resultToString(res), static_cast<int>(res));
            return false;
        }

        // Populate Vulkan-Hpp instance-level function pointers for NVRHI.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Instance(m_vkInstance));

        // Create persistent debug messenger to receive runtime validation logs beyond instance creation.
        if (m_validationEnabled) {
            auto createMessengerFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(m_vkInstance, "vkCreateDebugUtilsMessengerEXT")
            );
            if (createMessengerFn != nullptr) {
                (void)createMessengerFn(m_vkInstance, &debugInfo, nullptr, &m_debugMessenger);
            }
        }

        spdlog::info(
            "Vulkan {}.{} instance created (loader={}.{}.{}, validation={}).",
            VK_API_VERSION_MAJOR(m_instanceApiVersion),
            VK_API_VERSION_MINOR(m_instanceApiVersion),
            VK_API_VERSION_MAJOR(loaderVersion),
            VK_API_VERSION_MINOR(loaderVersion),
            VK_API_VERSION_PATCH(loaderVersion),
            m_validationEnabled ? "ON" : "OFF"
        );
        return true;
    }

    bool RenderDevice::CreateSurface() {
        const VkResult res = glfwCreateWindowSurface(
            m_vkInstance,
            m_window->GetNativeHandle(),
            nullptr,
            &m_vkSurface
        );
        if (res != VK_SUCCESS) {
            spdlog::error(
                "glfwCreateWindowSurface failed: {} ({}).",
                nvrhi::vulkan::resultToString(res),
                static_cast<int>(res)
            );
            return false;
        }
        spdlog::info("Vulkan presentation surface created.");
        return true;
    }

    bool RenderDevice::SelectPhysicalDevice(
        const core::HardwareCapabilities& caps,
        const core::HardwareAdaptationPlan& plan
    ) {
        std::uint32_t deviceCount = 0U;
        if (vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0U) {
            spdlog::error("No Vulkan-compatible physical GPUs found.");
            return false;
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, devices.data());

        // Locate the 16-byte hardware UUID of the primary CUDA GPU selected by SystemProfiler.
        const core::GPUDeviceInfo* targetCudaGpu = nullptr;
        for (const auto& gpu : caps.gpus) {
            if (gpu.deviceIndex == plan.primaryGpuIndex) {
                targetCudaGpu = &gpu;
                break;
            }
        }

        VkPhysicalDevice fallbackDevice = VK_NULL_HANDLE;
        std::uint32_t fallbackQueueFamily = 0U;
        std::string fallbackName;

        for (VkPhysicalDevice candidate : devices) {
            const std::optional<std::uint32_t> queueFamily =
                FindGraphicsPresentQueueFamily(candidate, m_vkSurface);
            if (!queueFamily.has_value()) {
                continue;
            }

            VkPhysicalDeviceIDProperties idProps{};
            idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &idProps;
            vkGetPhysicalDeviceProperties2(candidate, &props2);

            // Prefer exact 16-byte UUID match against the primary CUDA compute GPU.
            if (targetCudaGpu != nullptr &&
                std::memcmp(idProps.deviceUUID, targetCudaGpu->uuid.data(), VK_UUID_SIZE) == 0) {
                m_vkPhysicalDevice = candidate;
                m_graphicsQueueFamily = queueFamily.value();
                m_selectedPhysicalDeviceName = props2.properties.deviceName;
                m_matchedCudaUuid = true;
                spdlog::info(
                    "Matched Vulkan physical device '{}' directly to CUDA primary GPU [{}] via hardware UUID.",
                    m_selectedPhysicalDeviceName,
                    targetCudaGpu->deviceIndex
                );
                return true;
            }

            // Track the best discrete GPU as a fallback if UUID matching is unavailable.
            if (fallbackDevice == VK_NULL_HANDLE ||
                props2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                fallbackDevice = candidate;
                fallbackQueueFamily = queueFamily.value();
                fallbackName = props2.properties.deviceName;
            }
        }

        if (fallbackDevice == VK_NULL_HANDLE) {
            spdlog::error("No Vulkan GPU supports both graphics and presentation on the active surface.");
            return false;
        }

        m_vkPhysicalDevice = fallbackDevice;
        m_graphicsQueueFamily = fallbackQueueFamily;
        m_selectedPhysicalDeviceName = fallbackName;
        m_matchedCudaUuid = false;
        spdlog::warn(
            "Could not match CUDA GPU UUID to a Vulkan physical device; falling back to '{}'.",
            m_selectedPhysicalDeviceName
        );
        return true;
    }

    bool RenderDevice::CreateLogicalDevice() {
        const float queuePriority = 1.0f;

        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = m_graphicsQueueFamily;
        queueInfo.queueCount = 1U;
        queueInfo.pQueuePriorities = &queuePriority;

        m_enabledDeviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        // Clamp effective API version to what the selected physical GPU supports.
        VkPhysicalDeviceProperties selectedProps{};
        vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &selectedProps);
        const std::uint32_t deviceMajorMinor = VK_MAKE_API_VERSION(
            0U,
            VK_API_VERSION_MAJOR(selectedProps.apiVersion),
            VK_API_VERSION_MINOR(selectedProps.apiVersion),
            0U
        );
        m_effectiveApiVersion = std::min(m_instanceApiVersion, deviceMajorMinor);
        if (m_effectiveApiVersion < VK_API_VERSION_1_2) {
            spdlog::error(
                "Selected Vulkan GPU '{}' supports only {}.{}.{}; Vulkan >= 1.2 is required by NVRHI.",
                m_selectedPhysicalDeviceName,
                VK_API_VERSION_MAJOR(selectedProps.apiVersion),
                VK_API_VERSION_MINOR(selectedProps.apiVersion),
                VK_API_VERSION_PATCH(selectedProps.apiVersion)
            );
            return false;
        }

        // Enable Vulkan 1.2 timeline semaphores required internally by NVRHI's
        // Vulkan command-queue completion tracker, plus Vulkan 1.3 core
        // Synchronization2 & Dynamic Rendering when supported.
        VkPhysicalDeviceVulkan12Features supportedVk12Features{};
        supportedVk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

#if defined(VK_API_VERSION_1_3)
        VkPhysicalDeviceVulkan13Features supportedVk13Features{};
        supportedVk13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        if (m_effectiveApiVersion >= VK_API_VERSION_1_3) {
            supportedVk12Features.pNext = &supportedVk13Features;
        }
#endif

        VkPhysicalDeviceFeatures2 supportedFeatures2{};
        supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supportedFeatures2.pNext = &supportedVk12Features;
        vkGetPhysicalDeviceFeatures2(m_vkPhysicalDevice, &supportedFeatures2);

        if (supportedVk12Features.timelineSemaphore != VK_TRUE) {
            spdlog::error("Selected Vulkan GPU does not support Vulkan 1.2 timeline semaphores required by NVRHI.");
            return false;
        }

        VkPhysicalDeviceVulkan12Features enabledVk12Features{};
        enabledVk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        enabledVk12Features.timelineSemaphore = VK_TRUE;

#if defined(VK_API_VERSION_1_3)
        VkPhysicalDeviceVulkan13Features enabledVk13Features{};
        enabledVk13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        if (m_effectiveApiVersion >= VK_API_VERSION_1_3) {
            enabledVk13Features.synchronization2 = supportedVk13Features.synchronization2;
            enabledVk13Features.dynamicRendering = supportedVk13Features.dynamicRendering;
            enabledVk12Features.pNext = &enabledVk13Features;
        }
#endif

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.pNext = &enabledVk12Features;
        deviceInfo.queueCreateInfoCount = 1U;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(m_enabledDeviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames = m_enabledDeviceExtensions.data();

        const VkResult res = vkCreateDevice(m_vkPhysicalDevice, &deviceInfo, nullptr, &m_vkDevice);
        if (res != VK_SUCCESS) {
            spdlog::error("vkCreateDevice failed: {} ({}).", nvrhi::vulkan::resultToString(res), static_cast<int>(res));
            return false;
        }

        // Populate Vulkan-Hpp device-level function pointers for NVRHI.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk::Device(m_vkDevice));

        vkGetDeviceQueue(m_vkDevice, m_graphicsQueueFamily, 0U, &m_graphicsQueue);
        spdlog::info(
            "Logical Vulkan {}.{} device created (GPU supports {}.{}.{}, graphics/present queue family={}).",
            VK_API_VERSION_MAJOR(m_effectiveApiVersion),
            VK_API_VERSION_MINOR(m_effectiveApiVersion),
            VK_API_VERSION_MAJOR(selectedProps.apiVersion),
            VK_API_VERSION_MINOR(selectedProps.apiVersion),
            VK_API_VERSION_PATCH(selectedProps.apiVersion),
            m_graphicsQueueFamily
        );
        return true;
    }

    bool RenderDevice::CreateNvrhiDevice() {
        m_nvrhiMessageBridge = std::make_unique<NvrhiMessageBridge>();

        nvrhi::vulkan::DeviceDesc desc{};
        desc.errorCB = m_nvrhiMessageBridge.get();
        desc.instance = m_vkInstance;
        desc.physicalDevice = m_vkPhysicalDevice;
        desc.device = m_vkDevice;
        desc.graphicsQueue = m_graphicsQueue;
        desc.graphicsQueueIndex = static_cast<int>(m_graphicsQueueFamily);
        desc.instanceExtensions = m_enabledInstanceExtensions.data();
        desc.numInstanceExtensions = m_enabledInstanceExtensions.size();
        desc.deviceExtensions = m_enabledDeviceExtensions.data();
        desc.numDeviceExtensions = m_enabledDeviceExtensions.size();

        m_nvrhiDevice = nvrhi::vulkan::createDevice(desc);
        if (!m_nvrhiDevice) {
            spdlog::error("nvrhi::vulkan::createDevice returned null.");
            return false;
        }

        spdlog::info("NVIDIA NVRHI Vulkan device initialized.");
        return true;
    }

    bool RenderDevice::CreateSwapchainResources() {
        // ---------------------------------------------------------------------
        // 1. Query Surface Capabilities, Formats, and Compute Wayland-Safe Extent
        // ---------------------------------------------------------------------
        VkSurfaceCapabilitiesKHR surfCaps{};
        if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkPhysicalDevice, m_vkSurface, &surfCaps) != VK_SUCCESS) {
            spdlog::error("Failed to query Vulkan surface capabilities.");
            return false;
        }

        std::uint32_t formatCount = 0U;
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, m_vkSurface, &formatCount, nullptr);
        if (formatCount == 0U) {
            spdlog::error("Vulkan surface exposes zero supported pixel formats.");
            return false;
        }
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, m_vkSurface, &formatCount, formats.data());

        // Prefer 8-bit UNORM BGRA/RGBA with SRGB nonlinear colorspace so ImGui's
        // sRGB-authored colors render identically across Wayland, X11, and Win32.
        VkSurfaceFormatKHR chosenFormat = formats.front();
        for (const auto& fmt : formats) {
            if ((fmt.format == VK_FORMAT_B8G8R8A8_UNORM || fmt.format == VK_FORMAT_R8G8B8A8_UNORM) &&
                fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosenFormat = fmt;
                break;
            }
        }
        m_swapchainFormat = chosenFormat.format;
        m_swapchainColorSpace = chosenFormat.colorSpace;

        // On Wayland, `surfCaps.currentExtent` is `UINT32_MAX`, signaling that
        // the client's swapchain image size determines the mapped surface extent.
        if (surfCaps.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            m_swapchainExtent = surfCaps.currentExtent;
        } else {
            const std::uint32_t fbWidth = std::max(1U, m_window->GetFramebufferWidth());
            const std::uint32_t fbHeight = std::max(1U, m_window->GetFramebufferHeight());
            m_swapchainExtent.width = std::clamp(fbWidth, surfCaps.minImageExtent.width, surfCaps.maxImageExtent.width);
            m_swapchainExtent.height = std::clamp(fbHeight, surfCaps.minImageExtent.height, surfCaps.maxImageExtent.height);
        }

        if (m_swapchainExtent.width == 0U || m_swapchainExtent.height == 0U) {
            return false;
        }

        // Request triple buffering (minImageCount + 1) when permitted by the driver.
        const std::uint32_t maxImages = (surfCaps.maxImageCount > 0U) ? surfCaps.maxImageCount : 8U;
        m_swapchainMinImageCount = std::clamp(surfCaps.minImageCount + 1U, std::max(2U, surfCaps.minImageCount), maxImages);

        // ---------------------------------------------------------------------
        // 2. Create VkSwapchainKHR (FIFO VSync guaranteed on all Vulkan drivers)
        // ---------------------------------------------------------------------
        VkSwapchainCreateInfoKHR scInfo{};
        scInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        scInfo.surface = m_vkSurface;
        scInfo.minImageCount = m_swapchainMinImageCount;
        scInfo.imageFormat = m_swapchainFormat;
        scInfo.imageColorSpace = m_swapchainColorSpace;
        scInfo.imageExtent = m_swapchainExtent;
        scInfo.imageArrayLayers = 1U;
        scInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        scInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        scInfo.preTransform = surfCaps.currentTransform;
        scInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        scInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        scInfo.clipped = VK_TRUE;
        scInfo.oldSwapchain = VK_NULL_HANDLE;

        const VkResult scRes = vkCreateSwapchainKHR(m_vkDevice, &scInfo, nullptr, &m_swapchain);
        if (scRes != VK_SUCCESS) {
            spdlog::error("vkCreateSwapchainKHR failed: {} ({}).", nvrhi::vulkan::resultToString(scRes), static_cast<int>(scRes));
            return false;
        }

        // ---------------------------------------------------------------------
        // 3. Retrieve Swapchain Images, Create ImageViews & Wrap for NVRHI
        // ---------------------------------------------------------------------
        std::uint32_t actualImageCount = 0U;
        vkGetSwapchainImagesKHR(m_vkDevice, m_swapchain, &actualImageCount, nullptr);
        m_swapchainImages.resize(actualImageCount);
        vkGetSwapchainImagesKHR(m_vkDevice, m_swapchain, &actualImageCount, m_swapchainImages.data());

        m_swapchainImageViews.resize(actualImageCount, VK_NULL_HANDLE);
        m_swapchainNvrhiTextures.resize(actualImageCount);
        m_renderFinishedSemaphores.resize(actualImageCount, VK_NULL_HANDLE);

        const nvrhi::Format nvrhiSwapFormat = VkFormatToNvrhiFormat(m_swapchainFormat);

        for (std::size_t i = 0U; i < actualImageCount; ++i) {
            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = m_swapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_swapchainFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0U;
            viewInfo.subresourceRange.levelCount = 1U;
            viewInfo.subresourceRange.baseArrayLayer = 0U;
            viewInfo.subresourceRange.layerCount = 1U;

            if (vkCreateImageView(m_vkDevice, &viewInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
                spdlog::error("Failed to create swapchain VkImageView [{}].", i);
                return false;
            }

            // Wrap each native VkImage into an NVRHI TextureHandle so future
            // 2D/3D NVRHI simulation pipelines can target or blit to the swapchain.
            nvrhi::TextureDesc texDesc{};
            texDesc.dimension = nvrhi::TextureDimension::Texture2D;
            texDesc.width = m_swapchainExtent.width;
            texDesc.height = m_swapchainExtent.height;
            texDesc.format = nvrhiSwapFormat;
            texDesc.isRenderTarget = true;
            texDesc.initialState = nvrhi::ResourceStates::Present;
            texDesc.keepInitialState = true;
            texDesc.debugName = "UmgebungSwapchainImage";

            m_swapchainNvrhiTextures[i] = m_nvrhiDevice->createHandleForNativeTexture(
                nvrhi::ObjectTypes::VK_Image,
                nvrhi::Object(m_swapchainImages[i]),
                texDesc
            );

            // Create a dedicated render-finished semaphore per swapchain image.
            VkSemaphoreCreateInfo semInfo{};
            semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (vkCreateSemaphore(m_vkDevice, &semInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS) {
                spdlog::error("Failed to create per-image renderFinished semaphore [{}].", i);
                return false;
            }
        }

        // ---------------------------------------------------------------------
        // 4. Create Main Render Pass (Clear -> Color Attachment -> Present)
        // ---------------------------------------------------------------------
        if (m_renderPass == VK_NULL_HANDLE) {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = m_swapchainFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            VkAttachmentReference colorRef{};
            colorRef.attachment = 0U;
            colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1U;
            subpass.pColorAttachments = &colorRef;

            // External subpass dependency ensures color writes wait for image acquisition.
            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0U;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0U;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo rpInfo{};
            rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rpInfo.attachmentCount = 1U;
            rpInfo.pAttachments = &colorAttachment;
            rpInfo.subpassCount = 1U;
            rpInfo.pSubpasses = &subpass;
            rpInfo.dependencyCount = 1U;
            rpInfo.pDependencies = &dependency;

            if (vkCreateRenderPass(m_vkDevice, &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
                spdlog::error("Failed to create main Vulkan render pass.");
                return false;
            }
        }

        // ---------------------------------------------------------------------
        // 5. Create Per-Image Framebuffers
        // ---------------------------------------------------------------------
        m_swapchainFramebuffers.resize(actualImageCount, VK_NULL_HANDLE);
        for (std::size_t i = 0U; i < actualImageCount; ++i) {
            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass = m_renderPass;
            fbInfo.attachmentCount = 1U;
            fbInfo.pAttachments = &m_swapchainImageViews[i];
            fbInfo.width = m_swapchainExtent.width;
            fbInfo.height = m_swapchainExtent.height;
            fbInfo.layers = 1U;

            if (vkCreateFramebuffer(m_vkDevice, &fbInfo, nullptr, &m_swapchainFramebuffers[i]) != VK_SUCCESS) {
                spdlog::error("Failed to create swapchain VkFramebuffer [{}].", i);
                return false;
            }
        }

        return true;
    }

    void RenderDevice::DestroySwapchainResources() noexcept {
        if (m_vkDevice == VK_NULL_HANDLE) {
            return;
        }

        for (VkFramebuffer fb : m_swapchainFramebuffers) {
            if (fb != VK_NULL_HANDLE) {
                vkDestroyFramebuffer(m_vkDevice, fb, nullptr);
            }
        }
        m_swapchainFramebuffers.clear();

        m_swapchainNvrhiTextures.clear();

        for (VkSemaphore sem : m_renderFinishedSemaphores) {
            if (sem != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_vkDevice, sem, nullptr);
            }
        }
        m_renderFinishedSemaphores.clear();

        for (VkImageView view : m_swapchainImageViews) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(m_vkDevice, view, nullptr);
            }
        }
        m_swapchainImageViews.clear();
        m_swapchainImages.clear();

        if (m_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(m_vkDevice, m_swapchain, nullptr);
            m_swapchain = VK_NULL_HANDLE;
        }
    }

    bool RenderDevice::RecreateSwapchain() {
        if (m_window == nullptr || m_window->IsMinimized()) {
            return false;
        }

        vkDeviceWaitIdle(m_vkDevice);
        DestroySwapchainResources();

        if (!CreateSwapchainResources()) {
            spdlog::error("Failed to recreate Vulkan swapchain after resize.");
            return false;
        }

        if (m_imguiInitialized) {
            ImGui_ImplVulkan_SetMinImageCount(m_swapchainMinImageCount);
        }

        m_swapchainNeedsRecreate = false;
        spdlog::debug(
            "Recreated Vulkan swapchain ({}x{}, {} images).",
            m_swapchainExtent.width,
            m_swapchainExtent.height,
            m_swapchainImages.size()
        );
        return true;
    }

    bool RenderDevice::CreateFrameSyncObjects() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = m_graphicsQueueFamily;

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // Start fences signaled so the very first `BeginFrame()` does not deadlock waiting.
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (std::uint32_t i = 0U; i < kMaxFramesInFlight; ++i) {
            FrameSyncObjects& frame = m_frames[i];
            if (vkCreateCommandPool(m_vkDevice, &poolInfo, nullptr, &frame.commandPool) != VK_SUCCESS) {
                spdlog::error("Failed to create Vulkan command pool for frame slot {}.", i);
                return false;
            }

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = frame.commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1U;

            if (vkAllocateCommandBuffers(m_vkDevice, &allocInfo, &frame.commandBuffer) != VK_SUCCESS) {
                spdlog::error("Failed to allocate Vulkan command buffer for frame slot {}.", i);
                return false;
            }

            if (vkCreateSemaphore(m_vkDevice, &semInfo, nullptr, &frame.imageAvailableSemaphore) != VK_SUCCESS ||
                vkCreateFence(m_vkDevice, &fenceInfo, nullptr, &frame.inFlightFence) != VK_SUCCESS) {
                spdlog::error("Failed to create Vulkan sync primitives for frame slot {}.", i);
                return false;
            }
        }

        return true;
    }

    void RenderDevice::DestroyFrameSyncObjects() noexcept {
        if (m_vkDevice == VK_NULL_HANDLE) {
            return;
        }

        for (FrameSyncObjects& frame : m_frames) {
            if (frame.inFlightFence != VK_NULL_HANDLE) {
                vkDestroyFence(m_vkDevice, frame.inFlightFence, nullptr);
                frame.inFlightFence = VK_NULL_HANDLE;
            }
            if (frame.imageAvailableSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_vkDevice, frame.imageAvailableSemaphore, nullptr);
                frame.imageAvailableSemaphore = VK_NULL_HANDLE;
            }
            if (frame.commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(m_vkDevice, frame.commandPool, nullptr);
                frame.commandPool = VK_NULL_HANDLE;
                frame.commandBuffer = VK_NULL_HANDLE;
            }
        }
    }

    bool RenderDevice::InitializeImGui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
#if defined(UMGEBUNG_PLATFORM_LINUX)
        // Give detached multi-viewport panels native OS window decorations on
        // Linux so Wayland/XWayland compositors (e.g., KDE KWin) manage their
        // frame geometry and cursor coordinate mapping accurately.
        io.ConfigViewportsNoDecoration = false;
#endif

        ApplyUmgebungImGuiTheme();
        if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        if (!ImGui_ImplGlfw_InitForVulkan(m_window->GetNativeHandle(), true)) {
            spdlog::error("ImGui_ImplGlfw_InitForVulkan failed.");
            ImGui::DestroyContext();
            return false;
        }

        // Query HiDPI content scale AFTER `ImGui_ImplGlfw_InitForVulkan` so the
        // GLFW window -> ImGuiContext backend map is populated (required on Wayland).
        const float contentScale = ImGui_ImplGlfw_GetContentScaleForWindow(m_window->GetNativeHandle());
        if (contentScale > 0.0f && contentScale != 1.0f) {
            ImGui::GetStyle().ScaleAllSizes(contentScale);
            io.FontGlobalScale = contentScale;
        }

        ImGui_ImplVulkan_InitInfo initInfo{};
        initInfo.ApiVersion = m_effectiveApiVersion;
        initInfo.Instance = m_vkInstance;
        initInfo.PhysicalDevice = m_vkPhysicalDevice;
        initInfo.Device = m_vkDevice;
        initInfo.QueueFamily = m_graphicsQueueFamily;
        initInfo.Queue = m_graphicsQueue;
        initInfo.DescriptorPoolSize = 128U; // Covers main + secondary viewport font/texture descriptors.
        initInfo.MinImageCount = m_swapchainMinImageCount;
        initInfo.ImageCount = static_cast<std::uint32_t>(m_swapchainImages.size());
        initInfo.PipelineInfoMain.RenderPass = m_renderPass;
        initInfo.PipelineInfoMain.Subpass = 0U;
        initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.PipelineInfoForViewports.Subpass = 0U;
        initInfo.PipelineInfoForViewports.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        initInfo.CheckVkResultFn = CheckImGuiVkResult;

        if (!ImGui_ImplVulkan_Init(&initInfo)) {
            spdlog::error("ImGui_ImplVulkan_Init failed.");
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            return false;
        }

        m_imguiInitialized = true;
        const bool multiViewportActive =
            ((io.BackendFlags & ImGuiBackendFlags_PlatformHasViewports) != 0) &&
            ((io.BackendFlags & ImGuiBackendFlags_RendererHasViewports) != 0);
        spdlog::info(
            "Dear ImGui (Docking + GLFW + Vulkan backend) initialized (Multi-Viewport={}).",
            multiViewportActive ? "ACTIVE" : "UNAVAILABLE on current display backend"
        );
        return true;
    }

    void RenderDevice::ShutdownImGui() noexcept {
        if (!m_imguiInitialized) {
            return;
        }

        if (m_vkDevice != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_vkDevice);
        }

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_imguiInitialized = false;
    }

    bool RenderDevice::BeginFrame() {
        if (!m_initialized || m_window == nullptr || m_window->IsMinimized()) {
            return false;
        }

        // Rebuild swapchain whenever the OS/compositor signaled a resize or suboptimal state.
        if (m_swapchainNeedsRecreate || m_window->ConsumeResizeFlag()) {
            if (!RecreateSwapchain()) {
                return false;
            }
        }

        FrameSyncObjects& frame = m_frames[m_currentFrame];

        // Wait for the previous GPU submission occupying this frame slot to finish.
        vkWaitForFences(m_vkDevice, 1U, &frame.inFlightFence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());

        // Acquire the next presentable swapchain image index.
        const VkResult acquireRes = vkAcquireNextImageKHR(
            m_vkDevice,
            m_swapchain,
            std::numeric_limits<std::uint64_t>::max(),
            frame.imageAvailableSemaphore,
            VK_NULL_HANDLE,
            &m_currentImageIndex
        );

        if (acquireRes == VK_ERROR_OUT_OF_DATE_KHR) {
            m_swapchainNeedsRecreate = true;
            return false;
        }
        if (acquireRes != VK_SUCCESS && acquireRes != VK_SUBOPTIMAL_KHR) {
            spdlog::error(
                "vkAcquireNextImageKHR failed: {} ({}).",
                nvrhi::vulkan::resultToString(acquireRes),
                static_cast<int>(acquireRes)
            );
            return false;
        }

        if (acquireRes == VK_SUBOPTIMAL_KHR) {
            m_swapchainNeedsRecreate = true;
        }

        // Start a new Dear ImGui UI frame; caller may now emit ImGui windows/widgets.
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        m_frameActive = true;
        return true;
    }

    void RenderDevice::EndFrameAndPresent() {
        if (!m_frameActive) {
            return;
        }
        m_frameActive = false;

        // Finalize Dear ImGui draw lists into renderable vertex/index data.
        ImGui::Render();
        ImDrawData* drawData = ImGui::GetDrawData();

        FrameSyncObjects& frame = m_frames[m_currentFrame];

        // Reset fence and command buffer only after confirming we will submit work.
        vkResetFences(m_vkDevice, 1U, &frame.inFlightFence);
        vkResetCommandPool(m_vkDevice, frame.commandPool, 0U);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

        // Begin main render pass: clears the swapchain image and transitions
        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR automatically.
        VkClearValue clearValue{};
        clearValue.color.float32[0] = m_clearColor[0];
        clearValue.color.float32[1] = m_clearColor[1];
        clearValue.color.float32[2] = m_clearColor[2];
        clearValue.color.float32[3] = m_clearColor[3];

        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass = m_renderPass;
        rpBegin.framebuffer = m_swapchainFramebuffers[m_currentImageIndex];
        rpBegin.renderArea.offset = {0, 0};
        rpBegin.renderArea.extent = m_swapchainExtent;
        rpBegin.clearValueCount = 1U;
        rpBegin.pClearValues = &clearValue;

        vkCmdBeginRenderPass(frame.commandBuffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        // Record Dear ImGui draw calls into the active Vulkan render pass.
        ImGui_ImplVulkan_RenderDrawData(drawData, frame.commandBuffer);

        vkCmdEndRenderPass(frame.commandBuffer);
        vkEndCommandBuffer(frame.commandBuffer);

        // Submit recorded command buffer to the graphics queue.
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphore renderFinishedSem = m_renderFinishedSemaphores[m_currentImageIndex];

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1U;
        submitInfo.pWaitSemaphores = &frame.imageAvailableSemaphore;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1U;
        submitInfo.pCommandBuffers = &frame.commandBuffer;
        submitInfo.signalSemaphoreCount = 1U;
        submitInfo.pSignalSemaphores = &renderFinishedSem;

        const VkResult submitRes = vkQueueSubmit(m_graphicsQueue, 1U, &submitInfo, frame.inFlightFence);
        if (submitRes != VK_SUCCESS) {
            spdlog::error(
                "vkQueueSubmit failed: {} ({}).",
                nvrhi::vulkan::resultToString(submitRes),
                static_cast<int>(submitRes)
            );
            return;
        }

        // Present the rendered swapchain image to the OS compositor (Wayland / X11 / Win32).
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1U;
        presentInfo.pWaitSemaphores = &renderFinishedSem;
        presentInfo.swapchainCount = 1U;
        presentInfo.pSwapchains = &m_swapchain;
        presentInfo.pImageIndices = &m_currentImageIndex;

        const VkResult presentRes = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
        if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR) {
            m_swapchainNeedsRecreate = true;
        } else if (presentRes != VK_SUCCESS) {
            spdlog::error(
                "vkQueuePresentKHR failed: {} ({}).",
                nvrhi::vulkan::resultToString(presentRes),
                static_cast<int>(presentRes)
            );
        }

        // Update and render detached Dear ImGui multi-viewport OS windows
        // (supported on Windows Win32 and Linux X11/XWayland; safe no-op on Wayland).
        if ((ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        // Advance frame-in-flight ring index and reclaim deferred NVRHI resources.
        m_currentFrame = (m_currentFrame + 1U) % kMaxFramesInFlight;
        if (m_nvrhiDevice) {
            m_nvrhiDevice->runGarbageCollection();
        }
    }

    void RenderDevice::Shutdown() noexcept {
        if (m_vkDevice != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_vkDevice);
        }

        // 1. Shut down Dear ImGui before destroying render pass, swapchain, or device.
        ShutdownImGui();

        // 2. Destroy swapchain framebuffers, views, NVRHI texture wrappers, and swapchain.
        DestroySwapchainResources();

        if (m_vkDevice != VK_NULL_HANDLE && m_renderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(m_vkDevice, m_renderPass, nullptr);
            m_renderPass = VK_NULL_HANDLE;
        }

        // 3. Destroy per-frame command pools, semaphores, and fences.
        DestroyFrameSyncObjects();

        // 4. Release NVRHI device before destroying the underlying VkDevice.
        if (m_nvrhiDevice) {
            m_nvrhiDevice->waitForIdle();
            m_nvrhiDevice = nullptr;
        }
        m_nvrhiMessageBridge.reset();

        // 5. Destroy logical Vulkan device, presentation surface, debug messenger, and instance.
        if (m_vkDevice != VK_NULL_HANDLE) {
            vkDestroyDevice(m_vkDevice, nullptr);
            m_vkDevice = VK_NULL_HANDLE;
            m_graphicsQueue = VK_NULL_HANDLE;
        }

        if (m_vkInstance != VK_NULL_HANDLE && m_vkSurface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(m_vkInstance, m_vkSurface, nullptr);
            m_vkSurface = VK_NULL_HANDLE;
        }

        if (m_vkInstance != VK_NULL_HANDLE && m_debugMessenger != VK_NULL_HANDLE) {
            auto destroyMessengerFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(m_vkInstance, "vkDestroyDebugUtilsMessengerEXT")
            );
            if (destroyMessengerFn != nullptr) {
                destroyMessengerFn(m_vkInstance, m_debugMessenger, nullptr);
            }
            m_debugMessenger = VK_NULL_HANDLE;
        }

        if (m_vkInstance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_vkInstance, nullptr);
            m_vkInstance = VK_NULL_HANDLE;
        }

        m_vkPhysicalDevice = VK_NULL_HANDLE;
        m_window = nullptr;
        m_initialized = false;
        m_frameActive = false;
        m_swapchainNeedsRecreate = false;
        spdlog::info("RenderDevice shut down cleanly.");
    }

} // namespace umgebung::graphics
