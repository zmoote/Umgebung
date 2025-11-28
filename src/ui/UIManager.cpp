/**
 * @file UIManager.cpp
 * @brief Implements the UIManager class.
 */
#include "umgebung/ui/UIManager.hpp"

#include "umgebung/ui/imgui/ViewportPanel.hpp"
#include "umgebung/ui/imgui/HierarchyPanel.hpp"
#include "umgebung/ui/imgui/PropertiesPanel.hpp"
#include "umgebung/ui/imgui/AboutPanel.hpp"
#include "umgebung/ui/imgui/ConsolePanel.hpp"
#include "umgebung/ui/imgui/StatisticsPanel.hpp"
#include "umgebung/ui/imgui/FilePickerPanel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include "umgebung/scene/Scene.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include <imgui_internal.h>
#include <filesystem>

#include "umgebung/util/LogMacros.hpp"

#include "umgebung/scene/SceneSerializer.hpp"

namespace Umgebung::ui {

    UIManager::UIManager() = default;
    UIManager::~UIManager() = default;

    void UIManager::init(GLFWwindow* window, scene::Scene* scene, renderer::Framebuffer* framebuffer, renderer::Renderer* renderer, ecs::systems::DebugRenderSystem* debugRenderSystem) {
        scene_ = scene;
        m_Renderer = renderer;
        debugRenderSystem_ = debugRenderSystem;

        m_SceneSerializer = std::make_unique<scene::SceneSerializer>(scene_, m_Renderer);
        m_SceneSerializer->deserialize("assets/scenes/default.umgebung");

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

        // DPI Scaling Logic
        float xScale, yScale;
        glfwGetWindowContentScale(window, &xScale, &yScale);
        float dpiScale = xScale; // We typically just use the horizontal scale

        UMGEBUNG_LOG_INFO("The horizontal content scale is: {}", xScale);
        UMGEBUNG_LOG_INFO("The vertical content scale is: {}", yScale);

        if (dpiScale > 1.0f) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.ScaleAllSizes(dpiScale);

            io.FontGlobalScale = dpiScale;
        }

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 460");

        panels_.push_back(std::make_unique<imgui::ViewportPanel>(framebuffer));
        panels_.push_back(std::make_unique<imgui::HierarchyPanel>(scene_));
        panels_.push_back(std::make_unique<imgui::PropertiesPanel>(scene_, [this](const std::string& title, const std::string& buttonLabel, imgui::FilePickerPanel::FileSelectedCallback callback, const std::vector<std::string>& extensions) {
			filePickerPanel_->open(title, buttonLabel, callback, extensions);
		}));
        panels_.push_back(std::make_unique<imgui::ConsolePanel>());

        filePickerPanel_ = std::make_unique<imgui::FilePickerPanel>();
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

    void UIManager::setAppCallback(const AppCallbackFn& callback) {
        appCallback_ = callback;
    }

    void UIManager::setupDockspace() {
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
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
        ImGui::Begin("##MainHostWindow", nullptr, window_flags);
        ImGui::PopStyleVar(3);

        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
            ImGuiID dockspace_id = ImGui::GetID("UmgebungDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

            if (firstFrame_ && !std::filesystem::exists("imgui.ini")) {
                firstFrame_ = false;
                ImGui::DockBuilderRemoveNode(dockspace_id);
                ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->WorkSize);

                ImGuiID dock_main_id = dockspace_id;
                ImGuiID dock_right_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.20f, nullptr, &dock_main_id);
                ImGuiID dock_left_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.20f, nullptr, &dock_main_id);
                ImGuiID dock_bottom_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.25f, nullptr, &dock_main_id);

                ImGui::DockBuilderDockWindow("Hierarchy", dock_left_id);
                ImGui::DockBuilderDockWindow("Properties", dock_right_id);
                ImGui::DockBuilderDockWindow("Console", dock_bottom_id);
                ImGui::DockBuilderDockWindow("Viewport", dock_main_id);

                ImGui::DockBuilderFinish(dockspace_id);
            }
        }

        if (ImGui::BeginMenuBar()) {
                        if (ImGui::BeginMenu("File")) {
                            if (ImGui::MenuItem("Save Scene")) {
                                if (!currentScenePath_.empty()) {
                                    m_SceneSerializer->serialize(currentScenePath_);
                                } else {
                                    filePickerPanel_->open("Save Scene As...", "Save", [this](const std::filesystem::path& path) {
                                        currentScenePath_ = path;
                                        m_SceneSerializer->serialize(currentScenePath_);
                                    }, { ".umgebung" }, "assets/scenes");
                                }
                            }
                            if (ImGui::MenuItem("Save As...")) {
                                filePickerPanel_->open("Save Scene As...", "Save", [this](const std::filesystem::path& path) {
                                    currentScenePath_ = path;
                                    m_SceneSerializer->serialize(currentScenePath_);
                                }, { ".umgebung" }, "assets/scenes");
                            }
                            if (ImGui::MenuItem("Open Scene...")) {
                                filePickerPanel_->open("Open Scene", "Open", [this](const std::filesystem::path& path) {
                                    if (m_SceneSerializer->deserialize(path)) {
                                        currentScenePath_ = path;
                                    }
                                }, { ".umgebung" }, "assets/scenes");
                            }
                            ImGui::Separator();
                            if (ImGui::MenuItem("Exit")) {
                                if (appCallback_) {
                                    appCallback_();
                                }
                            }
                            ImGui::EndMenu();
                        }
            
            if (ImGui::BeginMenu("Tools")) {
                if (ImGui::MenuItem("Statistics")) {
                    if (auto* panel = getPanel<ui::imgui::StatisticsPanel>()) {
                        panel->open();
                    } else {
                        panels_.push_back(std::make_unique<imgui::StatisticsPanel>(debugRenderSystem_)); 
                    }
                }

                if (ImGui::MenuItem("Console")) {
                    if (auto* panel = getPanel<ui::imgui::ConsolePanel>()) {
                        panel->open();
                    }
                    else {
                        panels_.push_back(std::make_unique<imgui::ConsolePanel>());
                    }

                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View")) {
                if (ImGui::MenuItem("Hierarchy")) {
                    if (auto* panel = getPanel<ui::imgui::HierarchyPanel>()) { 
                        panel->open(); 
                    }
                    else { 
                        panels_.push_back(std::make_unique<imgui::HierarchyPanel>(scene_)); 
                    }
                }
                if (ImGui::MenuItem("Properties")) {
                    if (auto* panel = getPanel<ui::imgui::PropertiesPanel>()) { 
                        panel->open(); 
                    }
                    else { 
                        panels_.push_back(std::make_unique<imgui::PropertiesPanel>(scene_, [this](const std::string& title, const std::string& buttonLabel, imgui::FilePickerPanel::FileSelectedCallback callback, const std::vector<std::string>& extensions) {
						    filePickerPanel_->open(title, buttonLabel, callback, extensions);
						}));
                    }
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("About Umgebung")) {
                    if (auto* panel = getPanel<ui::imgui::AboutPanel>()) { 
                        panel->open(); 
                    } else { 
                        panels_.push_back(std::make_unique<imgui::AboutPanel>()); 
                    }

                }
                ImGui::EndMenu();
            }

            ImGui::EndMenuBar();
        }

        for (const auto& panel : panels_) {
            panel->onUIRender();
        }

        filePickerPanel_->onUIRender();

        ImGui::End();
    }

}