#include "umgebung/ui/UIManager.hpp"

// Required includes for panel creation and functionality
#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include "umgebung/ui/imgui/AboutPanel.hpp"
#include "umgebung/ui/imgui/AssetBrowserPanel.hpp"
#include "umgebung/ui/imgui/ConsolePanel.hpp"
#include "umgebung/ui/imgui/StatisticsPanel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include "umgebung/scene/Scene.hpp"

// Required includes for ImGui and GLFW
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

namespace Umgebung::ui {

    UIManager::UIManager() = default;
    UIManager::~UIManager() = default;

    void UIManager::init(GLFWwindow* window, scene::Scene* scene, renderer::Framebuffer* framebuffer) {
        scene_ = scene;

        // Initialize ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 460");

        // --- Create all of your panels ---
        panels_.push_back(std::make_unique<imgui::ViewportPanel>(framebuffer));

        // Panels that need the scene
        panels_.push_back(std::make_unique<imgui::HierarchyPanel>(scene));
        panels_.push_back(std::make_unique<imgui::PropertiesPanel>(scene));

        // Panels that DO NOT need the scene
        panels_.push_back(std::make_unique<imgui::ConsolePanel>());
        panels_.push_back(std::make_unique<imgui::StatisticsPanel>());
        panels_.push_back(std::make_unique<imgui::AssetBrowserPanel>());
        panels_.push_back(std::make_unique<imgui::AboutPanel>());
    }

    void UIManager::shutdown() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void UIManager::beginFrame() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void UIManager::endFrame() {
        setupDockspace();

        // Render all the panels
        for (const auto& panel : panels_) {
            panel->onUIRender();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }

    void UIManager::setupDockspace() {
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
            window_flags |= ImGuiWindowFlags_NoBackground;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
        }

        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Exit")) { /* Application::close(); */ }
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        ImGui::End();
    }

    imgui::ViewportPanel* UIManager::getViewportPanel() {
        for (const auto& panel : panels_) {
            if (auto* vp = dynamic_cast<imgui::ViewportPanel*>(panel.get())) {
                return vp;
            }
        }
        return nullptr;
    }

} // namespace Umgebung::ui