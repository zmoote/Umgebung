/**
 * @file RenderDevice.hpp
 * @brief GUI-exclusive Vulkan 1.4 (with automatic fallback down to 1.2) +
 *        NVIDIA NVRHI + Dear ImGui rendering device.
 *
 * @details
 * Owns the Vulkan graphics context (negotiating up to Vulkan 1.4 when supported
 * by the host loader and GPU, with a Vulkan 1.2 minimum baseline for NVRHI
 * timeline semaphores), CUDA-UUID-matched physical GPU binding, NVIDIA NVRHI
 * device wrapper (`nvrhi::vulkan::IDevice`), presentation swapchain, per-frame
 * GPU synchronization primitives, and the Dear ImGui Vulkan/GLFW UI renderer.
 *
 * Architectural boundaries:
 *   - `RenderDevice` borrows an already-initialized `umgebung::graphics::Window&`
 *     and never owns or destroys the OS window itself.
 *   - The caller (`Application`) MUST invoke `RenderDevice::Shutdown()` (or
 *     destroy `RenderDevice`) **before** `Window::Shutdown()` so the Vulkan
 *     swapchain and `VkSurfaceKHR` are released while the native window handle
 *     is still alive.
 *   - Compiled **exclusively** when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 *     Headless HPC builds never include or link this file.
 *
 * @note
 * Presenting a real Vulkan swapchain image via `EndFrameAndPresent()` also
 * fulfills the Wayland `xdg_toplevel` first-buffer commit requirement so native
 * Wayland desktop sessions (e.g., KDE Plasma KWin) map and display the window
 * immediately on frame 1.
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/graphics/RenderDevice.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/core/SystemProfiler.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <nvrhi/nvrhi.h>
#include <vulkan/vulkan_core.h>

namespace umgebung::graphics {

    class Window;

    /**
     * @struct RenderDeviceConfig
     * @brief Optional tuning parameters supplied to `RenderDevice::Initialize()`.
     */
    struct RenderDeviceConfig {
        /**
         * @brief Enables Khronos Vulkan validation layers (`VK_LAYER_KHRONOS_validation`)
         *        when supported by the host driver/SDK.
         *
         * @details
         * Defaults to `true` in Debug builds and `false` in Release builds to
         * catch API misuse during development without incurring validation
         * overhead in optimized runs.
         */
#if defined(NDEBUG)
        bool enableValidationLayers{false};
#else
        bool enableValidationLayers{true};
#endif

        /**
         * @brief RGBA clear color applied to the swapchain backbuffer at the
         *        start of each frame (deep-space navy default).
         */
        std::array<float, 4> clearColor{0.043f, 0.055f, 0.090f, 1.0f};
    };

    /**
     * @class RenderDevice
     * @brief Manages Vulkan 1.2 + NVRHI device state, presentation swapchain,
     *        frame synchronization, and Dear ImGui integration.
     */
    class RenderDevice final {
    public:
        /** @brief Number of CPU-prepared frames allowed in flight concurrently on the GPU. */
        static constexpr std::uint32_t kMaxFramesInFlight = 2U;

        /** @brief Constructs an uninitialized `RenderDevice`; call `Initialize()` before use. */
        RenderDevice();

        /** @brief Destructor; guarantees GPU idle wait and complete Vulkan/NVRHI/ImGui teardown. */
        ~RenderDevice();

        // Non-copyable and non-movable due to raw Vulkan handles and owned NVRHI/ImGui state.
        RenderDevice(const RenderDevice&) = delete;
        RenderDevice& operator=(const RenderDevice&) = delete;
        RenderDevice(RenderDevice&&) = delete;
        RenderDevice& operator=(RenderDevice&&) = delete;

        /**
         * @brief Initializes Vulkan 1.2, selects the physical GPU matching the
         *        adapted CUDA primary device UUID, creates the NVRHI device,
         *        builds the swapchain/render pass, and initializes Dear ImGui.
         *
         * @param window Already-initialized desktop `Window` providing the native surface.
         * @param caps   Startup hardware capabilities snapshot (provides GPU UUIDs).
         * @param plan   Startup hardware adaptation plan (provides `primaryGpuIndex`).
         * @param config Optional validation layer and clear-color configuration.
         *
         * @return `true` on complete initialization; `false` on any Vulkan,
         *         NVRHI, swapchain, or Dear ImGui failure (partial state is
         *         automatically rolled back via `Shutdown()`).
         *
         * @pre `window.IsInitialized() == true`.
         */
        [[nodiscard]] bool Initialize(
            Window& window,
            const core::HardwareCapabilities& caps,
            const core::HardwareAdaptationPlan& plan,
            const RenderDeviceConfig& config = {}
        );

        /**
         * @brief Waits for the GPU to finish all in-flight work and destroys
         *        Dear ImGui, swapchain, sync primitives, NVRHI device, and Vulkan
         *        objects in strict reverse creation order.
         *
         * @details
         * Idempotent; safe to invoke explicitly during `Application::Shutdown()`
         * and again inside `~RenderDevice()`.
         */
        void Shutdown() noexcept;

        /**
         * @brief Begins a new render frame: handles window minimization/resize,
         *        waits for the active frame fence, acquires the next swapchain
         *        image, and starts a fresh Dear ImGui frame (`ImGui::NewFrame()`).
         *
         * @return `true` if a valid swapchain image was acquired and the caller
         *         may record ImGui/scene draw commands followed by
         *         `EndFrameAndPresent()`; `false` if the frame should be skipped
         *         (e.g., window minimized or swapchain just recreated).
         */
        [[nodiscard]] bool BeginFrame();

        /**
         * @brief Finalizes Dear ImGui draw data, records the clear + ImGui
         *        render pass into the active command buffer, submits to the
         *        graphics queue, presents the swapchain image via
         *        `vkQueuePresentKHR`, and runs NVRHI garbage collection.
         *
         * @pre Must only be called when the immediately preceding `BeginFrame()`
         *      returned `true`.
         */
        void EndFrameAndPresent();

        /** @brief Updates the RGBA background clear color used on subsequent frames. */
        void SetClearColor(float r, float g, float b, float a = 1.0f) noexcept {
            m_clearColor = {r, g, b, a};
        }

        /** @brief Returns `true` if the Vulkan/NVRHI/ImGui device is initialized and ready. */
        [[nodiscard]] bool IsInitialized() const noexcept { return m_initialized; }

        /** @brief Returns the underlying NVRHI device handle for future 2D/3D simulation render passes. */
        [[nodiscard]] nvrhi::IDevice* GetNvrhiDevice() const noexcept { return m_nvrhiDevice.Get(); }

        /** @brief Returns the active swapchain pixel width. */
        [[nodiscard]] std::uint32_t GetSwapchainWidth() const noexcept { return m_swapchainExtent.width; }

        /** @brief Returns the active swapchain pixel height. */
        [[nodiscard]] std::uint32_t GetSwapchainHeight() const noexcept { return m_swapchainExtent.height; }

        /** @brief Returns the name of the selected Vulkan physical GPU. */
        [[nodiscard]] const std::string& GetSelectedVulkanGpuName() const noexcept {
            return m_selectedPhysicalDeviceName;
        }

        /** @brief Returns `true` if the Vulkan physical device UUID matched the adapted CUDA primary GPU. */
        [[nodiscard]] bool IsMatchedToCudaGpu() const noexcept { return m_matchedCudaUuid; }

        /** @brief Returns the active negotiated Vulkan API version (`VK_API_VERSION_1_4` down to `1_2`). */
        [[nodiscard]] std::uint32_t GetVulkanApiVersion() const noexcept { return m_effectiveApiVersion; }

    private:
        /**
         * @struct FrameSyncObjects
         * @brief Per-frame-in-flight Vulkan command buffer and GPU synchronization primitives.
         */
        struct FrameSyncObjects {
            /** @brief Command pool reset once per frame once `inFlightFence` signals. */
            VkCommandPool commandPool{VK_NULL_HANDLE};

            /** @brief Primary command buffer allocated from `commandPool`. */
            VkCommandBuffer commandBuffer{VK_NULL_HANDLE};

            /** @brief Binary semaphore signaled by `vkAcquireNextImageKHR` when image is ready. */
            VkSemaphore imageAvailableSemaphore{VK_NULL_HANDLE};

            /** @brief Host-visible fence ensuring CPU never overwrites an in-flight command buffer. */
            VkFence inFlightFence{VK_NULL_HANDLE};
        };

        /** @brief Private NVRHI diagnostic message bridge routing NVRHI errors to `spdlog`. */
        class NvrhiMessageBridge;

        [[nodiscard]] bool CreateVulkanInstance(const RenderDeviceConfig& config);
        [[nodiscard]] bool CreateSurface();
        [[nodiscard]] bool SelectPhysicalDevice(
            const core::HardwareCapabilities& caps,
            const core::HardwareAdaptationPlan& plan
        );
        [[nodiscard]] bool CreateLogicalDevice();
        [[nodiscard]] bool CreateNvrhiDevice();
        [[nodiscard]] bool CreateSwapchainResources();
        void DestroySwapchainResources() noexcept;
        [[nodiscard]] bool RecreateSwapchain();
        [[nodiscard]] bool CreateFrameSyncObjects();
        void DestroyFrameSyncObjects() noexcept;
        [[nodiscard]] bool InitializeImGui();
        void ShutdownImGui() noexcept;

        /** @brief Non-owning pointer to the desktop `Window` passed to `Initialize()`. */
        Window* m_window{nullptr};

        /** @brief Tracks full initialization state to keep `Shutdown()` idempotent. */
        bool m_initialized{false};

        /** @brief True if Vulkan validation layers were enabled and active on `m_vkInstance`. */
        bool m_validationEnabled{false};

        /** @brief True if `m_vkPhysicalDevice` matched the exact 16-byte UUID of the CUDA primary GPU. */
        bool m_matchedCudaUuid{false};

        /** @brief True while `BeginFrame()` has acquired an image awaiting `EndFrameAndPresent()`. */
        bool m_frameActive{false};

        /** @brief Sticky flag indicating swapchain must be rebuilt before the next frame. */
        bool m_swapchainNeedsRecreate{false};

        /** @brief Clear color applied at the beginning of each main render pass. */
        std::array<float, 4> m_clearColor{0.043f, 0.055f, 0.090f, 1.0f};

        /** @brief Human-readable name of the selected Vulkan physical GPU. */
        std::string m_selectedPhysicalDeviceName;

        // ---------------------------------------------------------------------
        // Core Vulkan Handles
        // ---------------------------------------------------------------------
        std::uint32_t m_instanceApiVersion{VK_API_VERSION_1_2};
        std::uint32_t m_effectiveApiVersion{VK_API_VERSION_1_2};
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkDebugUtilsMessengerEXT m_debugMessenger{VK_NULL_HANDLE};
        VkSurfaceKHR m_vkSurface{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        VkQueue m_graphicsQueue{VK_NULL_HANDLE};
        std::uint32_t m_graphicsQueueFamily{0U};

        /** @brief Enabled Vulkan instance extension names retained for NVRHI DeviceDesc. */
        std::vector<const char*> m_enabledInstanceExtensions;

        /** @brief Enabled Vulkan device extension names retained for NVRHI DeviceDesc. */
        std::vector<const char*> m_enabledDeviceExtensions;

        // ---------------------------------------------------------------------
        // NVIDIA NVRHI Device & Diagnostics
        // ---------------------------------------------------------------------
        std::unique_ptr<NvrhiMessageBridge> m_nvrhiMessageBridge;
        nvrhi::DeviceHandle m_nvrhiDevice;

        // ---------------------------------------------------------------------
        // Swapchain, Render Pass, Framebuffers & NVRHI Backbuffer Wrappers
        // ---------------------------------------------------------------------
        VkSwapchainKHR m_swapchain{VK_NULL_HANDLE};
        VkFormat m_swapchainFormat{VK_FORMAT_B8G8R8A8_UNORM};
        VkColorSpaceKHR m_swapchainColorSpace{VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        VkExtent2D m_swapchainExtent{0U, 0U};
        std::uint32_t m_swapchainMinImageCount{2U};

        std::vector<VkImage> m_swapchainImages;
        std::vector<VkImageView> m_swapchainImageViews;
        std::vector<nvrhi::TextureHandle> m_swapchainNvrhiTextures;
        std::vector<VkFramebuffer> m_swapchainFramebuffers;

        /**
         * @brief Per-swapchain-image binary semaphores signaled when rendering
         *        finishes and waited on by `vkQueuePresentKHR`.
         *
         * @details
         * Indexed by `m_currentImageIndex` (rather than `m_currentFrame`) so
         * presentation engines with 3+ swapchain images never reuse a semaphore
         * still pending in `vkQueuePresentKHR` (satisfies Vulkan validation
         * rule `VUID-vkQueueSubmit-pSignalSemaphores-00067`).
         */
        std::vector<VkSemaphore> m_renderFinishedSemaphores;

        VkRenderPass m_renderPass{VK_NULL_HANDLE};

        // ---------------------------------------------------------------------
        // Per-Frame-In-Flight Command Buffers & Sync Primitives
        // ---------------------------------------------------------------------
        std::array<FrameSyncObjects, kMaxFramesInFlight> m_frames{};
        std::uint32_t m_currentFrame{0U};
        std::uint32_t m_currentImageIndex{0U};

        // ---------------------------------------------------------------------
        // Dear ImGui State
        // ---------------------------------------------------------------------
        bool m_imguiInitialized{false};
    };

} // namespace umgebung::graphics
