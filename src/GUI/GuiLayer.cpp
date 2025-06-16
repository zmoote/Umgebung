#include "GuiLayer.hpp"
#include "../Renderer/Renderer.hpp"
#include "../Platform/Window.hpp"
#include "../Core/Logger.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace Umgebung {

    void GuiLayer::OnAttach(Renderer* pRenderer, Window* pWindow) {
        renderer = pRenderer;
        window = pWindow;

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable docking

        // Initialize GLFW backend
        if (!ImGui_ImplGlfw_InitForVulkan(window->GetNativeWindow(), true)) {
            Logger::GetCoreLogger()->error("Failed to initialize ImGui GLFW backend.");
            return;
        }

        // Create descriptor pool
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };
        VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000;
        pool_info.poolSizeCount = std::size(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        if (vkCreateDescriptorPool(renderer->get_logical_device(), &pool_info, nullptr, &descriptorPool) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to create ImGui descriptor pool.");
            return;
        }

        // Initialize Vulkan backend
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = renderer->get_instance();
        init_info.PhysicalDevice = renderer->get_physical_device();
        init_info.Device = renderer->get_logical_device();
        init_info.QueueFamily = renderer->get_graphics_queue_family();
        init_info.Queue = renderer->get_graphics_queue();
        init_info.DescriptorPool = descriptorPool;
        init_info.MinImageCount = 2;
        init_info.ImageCount = 3;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.RenderPass = renderer->get_render_pass();

        if (init_info.RenderPass == VK_NULL_HANDLE) {
            Logger::GetCoreLogger()->error("Render pass is null. Cannot initialize ImGui Vulkan backend.");
            return;
        }

        if (!ImGui_ImplVulkan_Init(&init_info)) {
            Logger::GetCoreLogger()->error("Failed to initialize ImGui Vulkan backend.");
            return;
        }

        // Upload fonts
        VkCommandPool commandPool;
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = renderer->get_graphics_queue_family();
        if (vkCreateCommandPool(renderer->get_logical_device(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to create command pool for ImGui fonts.");
            return;
        }

        VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VkCommandBuffer commandBuffer;
        if (vkAllocateCommandBuffers(renderer->get_logical_device(), &allocInfo, &commandBuffer) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to allocate command buffer for ImGui fonts.");
            vkDestroyCommandPool(renderer->get_logical_device(), commandPool, nullptr);
            return;
        }

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to begin command buffer for ImGui fonts.");
            vkFreeCommandBuffers(renderer->get_logical_device(), commandPool, 1, &commandBuffer);
            vkDestroyCommandPool(renderer->get_logical_device(), commandPool, nullptr);
            return;
        }

        ImGui_ImplVulkan_CreateFontsTexture();
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to end command buffer for ImGui fonts.");
            vkFreeCommandBuffers(renderer->get_logical_device(), commandPool, 1, &commandBuffer);
            vkDestroyCommandPool(renderer->get_logical_device(), commandPool, nullptr);
            return;
        }

        VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        if (vkQueueSubmit(renderer->get_graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            Logger::GetCoreLogger()->error("Failed to submit command buffer for ImGui fonts.");
            vkFreeCommandBuffers(renderer->get_logical_device(), commandPool, 1, &commandBuffer);
            vkDestroyCommandPool(renderer->get_logical_device(), commandPool, nullptr);
            return;
        }
        vkQueueWaitIdle(renderer->get_graphics_queue());

        vkFreeCommandBuffers(renderer->get_logical_device(), commandPool, 1, &commandBuffer);
        vkDestroyCommandPool(renderer->get_logical_device(), commandPool, nullptr);

        Logger::GetCoreLogger()->info("ImGui Layer attached.");
    }

    void GuiLayer::OnDetach() {
        vkDeviceWaitIdle(renderer->get_logical_device());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(renderer->get_logical_device(), descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }
        ImGui::DestroyContext();
        Logger::GetCoreLogger()->info("ImGui Layer detached.");
    }

    void GuiLayer::OnImGuiRender() {
        Logger::GetCoreLogger()->info("Rendering ImGui frame.");
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Enable docking (should also be set in OnAttach, but safe to set here)
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        // 2. Create a fullscreen dockspace window
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("DockSpace Demo", nullptr, window_flags);
        ImGui::PopStyleVar(2);

        // Dockspace
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

        // 3. Toolbar (top)
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Edit"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Tools"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Window"))
            {
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help"))
            {
                ImGui::MenuItem("About", nullptr, false, false); // Placeholder for "About" menu item
                ImGui::EndMenu();
			}
            ImGui::EndMainMenuBar();
        }

        // 4. Left panel
        ImGui::Begin("Left Panel");
        ImGui::Text("This is the left panel.");
        ImGui::End();

        // 5. Right panel
        ImGui::Begin("Right Panel");
        ImGui::Text("This is the right panel.");
        ImGui::End();

        ImGui::End(); // End DockSpace window

        // Optionally show the demo window
        // ImGui::ShowDemoWindow();

        ImGui::Render();
        ImDrawData* drawData = ImGui::GetDrawData();
        if (drawData == nullptr) {
            Logger::GetCoreLogger()->error("ImGui draw data is null.");
            return;
        }
        ImGui_ImplVulkan_RenderDrawData(drawData, renderer->get_current_command_buffer());
        Logger::GetCoreLogger()->info("ImGui frame rendered.");
    }

} // namespace Umgebung