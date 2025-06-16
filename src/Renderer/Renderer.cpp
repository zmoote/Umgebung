#include "Renderer.hpp"
#include "../Platform/Window.hpp"
#include "../Core/Logger.hpp"
#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

namespace Umgebung {

    namespace {
        constexpr VkClearColorValue DEFAULT_CLEAR_COLOR = { {0.0f, 0.0f, 0.0f, 1.0f} };
        constexpr uint64_t FENCE_TIMEOUT = UINT64_MAX;
    }

    // --- Private Helper Methods ---

    bool Renderer::IsWindowMinimized(Window* pWindow) const {
        int width, height;
        glfwGetFramebufferSize(pWindow->GetNativeWindow(), &width, &height);
        return width == 0 || height == 0;
    }

    void Renderer::HandleWindowResize(Window* pWindow) {
        if (pWindow->WasResized()) {
            Logger::GetCoreLogger()->warn("Window resized, triggering swapchain recreation.");
            framebufferResized = true;
            pWindow->ResetResizedFlag();
        }
    }

    bool Renderer::WaitAndResetFence() {
        VkResult waitResult = vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrameIndex], VK_TRUE, FENCE_TIMEOUT);
        if (waitResult != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to wait for fence for frame {} (VkResult: {}).", currentFrameIndex, static_cast<int>(waitResult));
            return false;
        }
        VkResult resetResult = vkResetFences(logicalDevice, 1, &inFlightFences[currentFrameIndex]);
        if (resetResult != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to reset fence for frame {} (VkResult: {}).", currentFrameIndex, static_cast<int>(resetResult));
            return false;
        }
        return true;
    }

    bool Renderer::AcquireNextSwapchainImage() {
        VkResult result = vkAcquireNextImageKHR(
            logicalDevice,
            swapchain,
            FENCE_TIMEOUT,
            imageAvailableSemaphores[currentFrameIndex],
            VK_NULL_HANDLE,
            &currentImageIndex
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            Logger::GetCoreLogger()->info("Swapchain out of date or resized, recreating.");
            framebufferResized = false;
            if (!isRecreatingSwapchain) {
                RecreateSwapchain();
            }
            return false;
        }
        if (result != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to acquire swapchain image (VkResult: {}).", static_cast<int>(result));
            return false;
        }
        return true;
    }

    bool Renderer::BeginCommandBuffer(VkCommandBuffer commandBuffer) {
        VkResult resetResult = vkResetCommandBuffer(commandBuffer, 0);
        if (resetResult != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to reset command buffer for image index {} (VkResult: {}).", currentImageIndex, static_cast<int>(resetResult));
            return false;
        }
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkResult beginResult = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        if (beginResult != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to begin command buffer for image index {} (VkResult: {}).", currentImageIndex, static_cast<int>(beginResult));
            return false;
        }
        return true;
    }

    void Renderer::BeginRenderPass(VkCommandBuffer commandBuffer) {
        VkClearValue clearColor = { DEFAULT_CLEAR_COLOR };
        VkRenderPassBeginInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[currentImageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapchainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        Logger::GetCoreLogger()->debug("Render pass begun for image index {}.", currentImageIndex);
    }

    void Renderer::SubmitFrame(VkCommandBuffer commandBuffer) {
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentImageIndex] };
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentImageIndex] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        VkResult submitResult = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrameIndex]);
        if (submitResult != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to submit draw command buffer for frame {} (VkResult: {}).", currentFrameIndex, static_cast<int>(submitResult));
        }
    }

    void Renderer::PresentFrame() {
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentImageIndex] };
        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &currentImageIndex;

        VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            Logger::GetCoreLogger()->warn("Swapchain out of date, triggering recreation.");
            framebufferResized = true;
        }
        else if (result != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to present swapchain image (VkResult: {}).", static_cast<int>(result));
        }
    }

    // --- Refactored Public Methods ---

    void Renderer::Init(Window* pWindow) {
        this->window = pWindow;
        Logger::GetCoreLogger()->info("Initializing Vulkan...");
        InitVulkan(pWindow);
        CreateSwapchain();
        CreateRenderPass();
        CreateFramebuffers();
        CreateCommandBuffers();
        InitSyncObjects();
        Logger::GetCoreLogger()->info("Vulkan renderer initialized successfully.");
    }

    bool Renderer::BeginFrame(Window* pWindow) {
        Logger::GetCoreLogger()->debug("BeginFrame: imageIndex {}", currentImageIndex);

        if (IsWindowMinimized(pWindow)) {
            Logger::GetCoreLogger()->debug("Window minimized, skipping frame.");
            glfwPollEvents();
            return false;
        }

        HandleWindowResize(pWindow);

        if (!WaitAndResetFence())
            return false;

        if (!AcquireNextSwapchainImage())
            return false;

        VkCommandBuffer commandBuffer = commandBuffers[currentImageIndex];
        if (commandBuffer == VK_NULL_HANDLE) {
            Logger::GetCoreLogger()->error("Invalid command buffer for image index {}.", currentImageIndex);
            return false;
        }

        if (!BeginCommandBuffer(commandBuffer))
            return false;

        BeginRenderPass(commandBuffer);
        return true;
    }

    void Renderer::EndFrame() {
        Logger::GetCoreLogger()->debug("EndFrame: frameIndex {}", currentFrameIndex);
        VkCommandBuffer commandBuffer = commandBuffers[currentImageIndex];
        if (commandBuffer == VK_NULL_HANDLE) {
            Logger::GetCoreLogger()->error("Invalid command buffer for image index {}.", currentImageIndex);
            return;
        }

        vkCmdEndRenderPass(commandBuffer);
        VkResult endResult = vkEndCommandBuffer(commandBuffer);
        if (endResult != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to end command buffer for image index {} (VkResult: {}).", currentImageIndex, static_cast<int>(endResult));
            return;
        }

        SubmitFrame(commandBuffer);
        PresentFrame();

        currentFrameIndex = (currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
        Logger::GetCoreLogger()->debug("Frame completed: next frameIndex {}.", currentFrameIndex);
    }

    void Renderer::Cleanup() {
        Logger::GetCoreLogger()->info("Cleaning up Vulkan...");

        if (logicalDevice != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(logicalDevice);
        }

        CleanupSyncObjects();

        // --- Begin: Destroy all tracked buffers, images, and device memory before destroying device ---
        if (!buffers.empty()) {
            Logger::GetCoreLogger()->warn("Buffers not destroyed before device destruction. Count: {}", buffers.size());
        }
        for (auto buffer : buffers) {
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(logicalDevice, buffer, nullptr);
                Logger::GetCoreLogger()->debug("Destroyed buffer: {}", reinterpret_cast<uint64_t>(buffer));
                UntrackBuffer(buffer);
            }
        }
        buffers.clear();

        if (!images.empty()) {
            Logger::GetCoreLogger()->warn("Images not destroyed before device destruction. Count: {}", images.size());
        }
        for (auto image : images) {
            if (image != VK_NULL_HANDLE) {
                vkDestroyImage(logicalDevice, image, nullptr);
                Logger::GetCoreLogger()->debug("Destroyed image: {}", reinterpret_cast<uint64_t>(image));
                UntrackImage(image);
            }
        }
        images.clear();

        if (!memories.empty()) {
            Logger::GetCoreLogger()->warn("Device memory not freed before device destruction. Count: {}", memories.size());
        }
        for (auto memory : memories) {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(logicalDevice, memory, nullptr);
                Logger::GetCoreLogger()->debug("Freed device memory: {}", reinterpret_cast<uint64_t>(memory));
                UntrackMemory(memory);
            }
        }
        memories.clear();
        // --- End: Resource cleanup ---

        auto safeDestroy = [this](auto destroyFunc, auto handle, const char* name) {
            if (handle != VK_NULL_HANDLE) {
                destroyFunc(logicalDevice, handle, nullptr);
                Logger::GetCoreLogger()->debug("Destroyed {}.", name);
            }
        };

        for (auto framebuffer : framebuffers) {
            safeDestroy(vkDestroyFramebuffer, framebuffer, "framebuffer");
        }
        framebuffers.clear();

        for (auto view : swapchainImageViews) {
            safeDestroy(vkDestroyImageView, view, "image view");
        }
        swapchainImageViews.clear();
        swapchainImages.clear();

        if (!commandBuffers.empty() && commandPool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(logicalDevice, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
            commandBuffers.clear();
            Logger::GetCoreLogger()->debug("Freed command buffers.");
        }

        safeDestroy(vkDestroyRenderPass, renderPass, "render pass");
        renderPass = VK_NULL_HANDLE;

        safeDestroy(vkDestroySwapchainKHR, swapchain, "swapchain");
        swapchain = VK_NULL_HANDLE;

        safeDestroy(vkDestroyCommandPool, commandPool, "command pool");
        commandPool = VK_NULL_HANDLE;

        if (surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance.instance, surface, nullptr);
            Logger::GetCoreLogger()->debug("Destroyed surface.");
            surface = VK_NULL_HANDLE;
        }

        if (logicalDevice != VK_NULL_HANDLE) {
            vkDestroyDevice(logicalDevice, nullptr);
            Logger::GetCoreLogger()->debug("Destroyed logical device.");
            logicalDevice = VK_NULL_HANDLE;
        }

        vkb::destroy_instance(instance);
        Logger::GetCoreLogger()->info("Vulkan cleanup complete.");
    }

    // --- MISSING METHOD DEFINITIONS ---

    void Renderer::InitVulkan(Window* pWindow) {
        vkb::InstanceBuilder builder;
        auto inst_ret = builder.set_app_name("Umgebung")
            .request_validation_layers(true)
            .use_default_debug_messenger()
            .require_api_version(1, 3, 0)
            .build();
        if (!inst_ret) {
            Logger::GetCoreLogger()->error("Failed to create Vulkan instance: {}", inst_ret.error().message());
            std::terminate();
        }
        instance = inst_ret.value();

        if (glfwCreateWindowSurface(instance.instance, pWindow->GetNativeWindow(), nullptr, &surface) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to create window surface.");
            std::terminate();
        }

        vkb::PhysicalDeviceSelector selector{ instance };
        auto phys_ret = selector.set_surface(surface).select();
        if (!phys_ret) {
            Logger::GetCoreLogger()->error("Failed to select physical device: {}", phys_ret.error().message());
            std::terminate();
        }
        physicalDevice = phys_ret.value().physical_device;

        vkb::DeviceBuilder deviceBuilder{ phys_ret.value() };
        auto dev_ret = deviceBuilder.build();
        if (!dev_ret) {
            Logger::GetCoreLogger()->error("Failed to create logical device: {}", dev_ret.error().message());
            std::terminate();
        }
        device = dev_ret.value();
        logicalDevice = device.device;

        graphicsQueue = device.get_queue(vkb::QueueType::graphics).value();
        graphicsQueueFamily = device.get_queue_index(vkb::QueueType::graphics).value();
        presentQueue = device.get_queue(vkb::QueueType::present).value();
    }

    void Renderer::CreateSwapchain() {
        int width, height;
        glfwGetFramebufferSize(window->GetNativeWindow(), &width, &height);
        if (width == 0 || height == 0) {
            Logger::GetCoreLogger()->warn("Attempted to create swapchain with 0x0 dimensions.");
            return;
        }

        vkb::SwapchainBuilder swapchainBuilder{ device };
        auto swap_ret = swapchainBuilder
            .set_old_swapchain(swapchain)
            .set_desired_format({ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
            .set_desired_extent(width, height)
            .build();
        if (!swap_ret) {
            Logger::GetCoreLogger()->error("Failed to create swapchain: {}", swap_ret.error().message());
            std::terminate();
        }

        auto swapchainBundle = swap_ret.value();
        swapchain = swapchainBundle.swapchain;
        swapchainImages = swapchainBundle.get_images().value();
        swapchainImageViews = swapchainBundle.get_image_views().value();
        swapchainImageFormat = swapchainBundle.image_format;
        swapchainExtent = swapchainBundle.extent;
    }

    void Renderer::CreateRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to create render pass.");
            std::terminate();
        }
    }

    void Renderer::CreateFramebuffers() {
        framebuffers.resize(swapchainImageViews.size());
        for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
            VkImageView attachments[] = { swapchainImageViews[i] };
            VkFramebufferCreateInfo framebufferInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapchainExtent.width;
            framebufferInfo.height = swapchainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create framebuffer for image {}", i);
                std::terminate();
            }
        }
    }

    void Renderer::CreateCommandBuffers() {
        if (commandPool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            poolInfo.queueFamilyIndex = graphicsQueueFamily;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create command pool.");
                std::terminate();
            }
        }

        commandBuffers.resize(swapchainImages.size());
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to allocate command buffers.");
            std::terminate();
        }
    }

    void Renderer::InitSyncObjects() {
        size_t imageCount = swapchainImages.size();
        imageAvailableSemaphores.resize(imageCount);
        renderFinishedSemaphores.resize(imageCount);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < imageCount; ++i) {
            if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create synchronization semaphores for image {}", i);
                std::terminate();
            }
        }
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create fence for frame {}", i);
                std::terminate();
            }
        }
    }

    void Renderer::RecreateSwapchain() {
        vkDeviceWaitIdle(logicalDevice);

        // Destroy old framebuffers and image views
        for (auto framebuffer : framebuffers) {
            if (framebuffer != VK_NULL_HANDLE)
                vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
        }
        framebuffers.clear();

        for (auto view : swapchainImageViews) {
            if (view != VK_NULL_HANDLE)
                vkDestroyImageView(logicalDevice, view, nullptr);
        }
        swapchainImageViews.clear();
        swapchainImages.clear();

        if (swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(logicalDevice, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }

        // Destroy and recreate sync objects because their count depends on swapchain image count
        CleanupSyncObjects();

        CreateSwapchain();
        CreateFramebuffers();
        CreateCommandBuffers();

        // Recreate sync objects for new swapchain image count
        InitSyncObjects();
    }

    void Renderer::CleanupSyncObjects() {
        for (size_t i = 0; i < imageAvailableSemaphores.size(); ++i) {
            if (imageAvailableSemaphores[i] != VK_NULL_HANDLE)
                vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
            if (renderFinishedSemaphores[i] != VK_NULL_HANDLE)
                vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
        }
        imageAvailableSemaphores.clear();
        renderFinishedSemaphores.clear();

        for (size_t i = 0; i < inFlightFences.size(); ++i) {
            if (inFlightFences[i] != VK_NULL_HANDLE)
                vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
        }
        inFlightFences.clear();
    }

    // --- Resource tracking implementations ---
    void Renderer::TrackBuffer(VkBuffer buffer) {
        if (buffer != VK_NULL_HANDLE) {
            // Prevent double-tracking
            if (std::find(buffers.begin(), buffers.end(), buffer) == buffers.end()) {
                buffers.push_back(buffer);
                Logger::GetCoreLogger()->debug("Tracked buffer: {}", reinterpret_cast<uint64_t>(buffer));
            }
        }
    }
    void Renderer::TrackImage(VkImage image) {
        if (image != VK_NULL_HANDLE) {
            if (std::find(images.begin(), images.end(), image) == images.end()) {
                images.push_back(image);
                Logger::GetCoreLogger()->debug("Tracked image: {}", reinterpret_cast<uint64_t>(image));
            }
        }
    }
    void Renderer::TrackMemory(VkDeviceMemory memory) {
        if (memory != VK_NULL_HANDLE) {
            if (std::find(memories.begin(), memories.end(), memory) == memories.end()) {
                memories.push_back(memory);
                Logger::GetCoreLogger()->debug("Tracked device memory: {}", reinterpret_cast<uint64_t>(memory));
            }
        }
    }

    // --- Resource untracking implementations ---
    void Renderer::UntrackBuffer(VkBuffer buffer) {
        auto it = std::find(buffers.begin(), buffers.end(), buffer);
        if (it != buffers.end()) {
            buffers.erase(it);
            Logger::GetCoreLogger()->debug("Untracked buffer: {}", reinterpret_cast<uint64_t>(buffer));
        }
    }
    void Renderer::UntrackImage(VkImage image) {
        auto it = std::find(images.begin(), images.end(), image);
        if (it != images.end()) {
            images.erase(it);
            Logger::GetCoreLogger()->debug("Untracked image: {}", reinterpret_cast<uint64_t>(image));
        }
    }
    void Renderer::UntrackMemory(VkDeviceMemory memory) {
        auto it = std::find(memories.begin(), memories.end(), memory);
        if (it != memories.end()) {
            memories.erase(it);
            Logger::GetCoreLogger()->debug("Untracked device memory: {}", reinterpret_cast<uint64_t>(memory));
        }
    }

}