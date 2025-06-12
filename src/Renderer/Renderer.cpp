#include "Renderer.hpp"
#include "../Platform/Window.hpp"
#include "../Core/Logger.hpp"

#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

namespace Umgebung {

    void Renderer::Init(Window* window) {
        Logger::GetCoreLogger()->info("Initializing Vulkan...");
        InitVulkan(window);
        CreateSwapchain();
        CreateRenderPass();
        CreateFramebuffers();
        CreateCommandBuffers();
        InitSyncObjects();
    }

    bool Renderer::BeginFrame(Window* window) {
        vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrameIndex], VK_TRUE, UINT64_MAX);
        vkResetFences(logicalDevice, 1, &inFlightFences[currentFrameIndex]);

        VkResult result = vkAcquireNextImageKHR(
            logicalDevice,
            swapchain,
            UINT64_MAX,
            imageAvailableSemaphores[currentFrameIndex],
            VK_NULL_HANDLE,
            &currentImageIndex
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            Logger::GetCoreLogger()->warn("Swapchain out of date or suboptimal — recreating.");
            RecreateSwapchain();
            return false;
        }

        if (result != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to acquire swapchain image.");
            return false;
        }

        // Begin command buffer recording
        VkCommandBuffer commandBuffer = commandBuffers[currentImageIndex];

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkResetCommandBuffer(commandBuffer, 0);
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // Start render pass
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };

        VkRenderPassBeginInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[currentImageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapchainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        // [Draw commands will go here later...]

        return true;
    }

    void Renderer::EndFrame() {
        VkCommandBuffer commandBuffer = commandBuffers[currentImageIndex];

        vkCmdEndRenderPass(commandBuffer);
        vkEndCommandBuffer(commandBuffer);

        // Submit
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrameIndex] };
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrameIndex] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrameIndex]) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to submit draw command buffer.");
            return;
        }

        // Present
        VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &currentImageIndex;
        presentInfo.pResults = nullptr;

        vkQueuePresentKHR(presentQueue, &presentInfo);

        currentFrameIndex = (currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void Renderer::Cleanup() {
        Logger::GetCoreLogger()->info("Cleaning up Vulkan...");

        // 1. Wait for device to be idle before destroying anything
        if (logicalDevice != VK_NULL_HANDLE)
            vkDeviceWaitIdle(logicalDevice);

        // 2. Destroy sync objects before destroying the device!
        CleanupSyncObjects();  // <- Must come BEFORE vkDestroyDevice()

        // 3. Destroy framebuffers, image views, swapchain, etc.
        for (auto framebuffer : framebuffers)
            if (framebuffer != VK_NULL_HANDLE)
                vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
        framebuffers.clear();

        for (auto view : swapchainImageViews)
            if (view != VK_NULL_HANDLE)
                vkDestroyImageView(logicalDevice, view, nullptr);
        swapchainImageViews.clear();
        swapchainImages.clear();

        if (renderPass != VK_NULL_HANDLE)
            vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        if (swapchain != VK_NULL_HANDLE)
            vkDestroySwapchainKHR(logicalDevice, swapchain, nullptr);

        if (commandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

        // 4. Finally, destroy the device itself
        if (logicalDevice != VK_NULL_HANDLE)
            vkDestroyDevice(logicalDevice, nullptr);
        logicalDevice = VK_NULL_HANDLE;

        // 5. Destroy Vulkan instance
        if (instance.instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance.instance, nullptr);
            instance = {};  // Safely reset
        }
    }

    void Renderer::InitVulkan(Window* window) {

        vkb::InstanceBuilder builder;
        auto inst_ret = builder.set_app_name("Umgebung")
            .request_validation_layers(ENABLE_VULKAN_VALIDATION_LAYERS)
            .use_default_debug_messenger()
            .require_api_version(1, 3, 0)
            .build();

        if (!inst_ret) {
            Logger::GetCoreLogger()->error("Failed to create Vulkan instance: {}", inst_ret.error().message());
            return;
        }

        bootstrapInstance = inst_ret.value();
        instance = bootstrapInstance;
        debugMessenger = bootstrapInstance.debug_messenger;

        if (glfwCreateWindowSurface(instance.instance, window->GetNativeWindow(), nullptr, &surface) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to create window surface.");
            return;
        }

        vkb::PhysicalDeviceSelector selector{ instance };
        auto phys_ret = selector.set_surface(surface).select();

        if (!phys_ret) {
            Logger::GetCoreLogger()->error("Failed to select physical device: {}", phys_ret.error().message());
            return;
        }

        vkb::DeviceBuilder deviceBuilder{ phys_ret.value() };
        auto dev_ret = deviceBuilder.build();

        if (!dev_ret) {
            Logger::GetCoreLogger()->error("Failed to create logical device: {}", dev_ret.error().message());
            return;
        }

        bootstrapDevice = dev_ret.value();
        device = bootstrapDevice;
        logicalDevice = device.device;
        physicalDevice = phys_ret.value().physical_device;

        graphicsQueue = bootstrapDevice.get_queue(vkb::QueueType::graphics).value();
        graphicsQueueFamily = bootstrapDevice.get_queue_index(vkb::QueueType::graphics).value();

        presentQueue = bootstrapDevice.get_queue(vkb::QueueType::present).value();
        assert(presentQueue != VK_NULL_HANDLE && "Failed to retrieve present queue!");

        Logger::GetCoreLogger()->info("Vulkan successfully initialized.");
    }

    void Renderer::CreateSwapchain() {
        vkb::SwapchainBuilder swapchainBuilder{ bootstrapDevice };

        auto swap_ret = swapchainBuilder
            .set_old_swapchain(swapchain)
            .set_desired_format({ VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
            .build();

        if (!swap_ret) {
            Logger::GetCoreLogger()->error("Failed to create swapchain: {}", swap_ret.error().message());
            return;
        }

        auto swapchainBundle = swap_ret.value();
        swapchain = swapchainBundle.swapchain;
        swapchainImages = swapchainBundle.get_images().value();
        swapchainImageViews = swapchainBundle.get_image_views().value();
        swapchainImageFormat = swapchainBundle.image_format;
        swapchainExtent = swapchainBundle.extent;

        Logger::GetCoreLogger()->info("Swapchain created: {}x{}, Format: {}",
            swapchainExtent.width,
            swapchainExtent.height,
            static_cast<int>(swapchainImageFormat));
    }

    void Renderer::CreateRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;      // Clear at beginning
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;    // Store result for presentation
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // Present to screen

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to create render pass.");
        }
        else {
            Logger::GetCoreLogger()->info("Render pass created.");
        }
    }

    void Renderer::CreateFramebuffers() {
        framebuffers.resize(swapchainImageViews.size());

        for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
            VkImageView attachments[] = {
                swapchainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapchainExtent.width;
            framebufferInfo.height = swapchainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create framebuffer for image {}", i);
            }
            else {
                Logger::GetCoreLogger()->info("Framebuffer {} created.", i);
            }
        }
    }

    void Renderer::CreateCommandBuffers() {
        // Create a command pool if not already created
        if (commandPool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = graphicsQueueFamily;

            if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create command pool.");
                return;
            }
            Logger::GetCoreLogger()->info("Command pool created.");
        }

        // Allocate command buffers
        commandBuffers.resize(framebuffers.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to allocate command buffers.");
            return;
        }

        Logger::GetCoreLogger()->info("Allocated {} command buffers.", commandBuffers.size());
    }

    void Renderer::InitSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                Logger::GetCoreLogger()->error("Failed to create sync objects for frame {}", i);
            }
        }
    }

    void Renderer::RecreateSwapchain() {
        Logger::GetCoreLogger()->info("Recreating swapchain...");

        // Wait for GPU to be idle
        vkDeviceWaitIdle(logicalDevice);

        // Cleanup old resources (order matters!)
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

        if (renderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
            renderPass = VK_NULL_HANDLE;
        }

        if (swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(logicalDevice, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }

        // Recreate everything
        CreateSwapchain();
        CreateRenderPass();
        CreateFramebuffers();
        CreateCommandBuffers();

        CleanupSyncObjects();

        InitSyncObjects();  // <-- ADD THIS LINE

        Logger::GetCoreLogger()->info("Swapchain recreation complete.");
    }

    void Renderer::CleanupSyncObjects() {

        if (imageAvailableSemaphores.empty()) return;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (imageAvailableSemaphores[i] != VK_NULL_HANDLE)
                vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
            if (renderFinishedSemaphores[i] != VK_NULL_HANDLE)
                vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
            if (inFlightFences[i] != VK_NULL_HANDLE)
                vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
        }

        imageAvailableSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
    }

} // namespace Umgebung