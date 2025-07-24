#include "../../include/walbourn/pch.h"
#include "../../include/umgebung/Simulation.h"
#include <chrono>

extern void ExitSimulation() noexcept;

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace Umgebung {

    Simulation::Simulation() noexcept(false)
        : m_imguiInitialized(false)
        , m_windowHandle(nullptr)
        , m_lastMouseX(0.0f)
        , m_lastMouseY(0.0f)
        , m_firstMouse(true)
        , m_mouseCaptured(false)
    {
        m_deviceResources = std::make_unique<DX::DeviceResources>();
        m_deviceResources->RegisterDeviceNotify(this);

        // Initialize input state
        memset(m_keys, 0, sizeof(m_keys));
        memset(m_mouseButtons, 0, sizeof(m_mouseButtons));

        // Initialize camera system
        m_camera = std::make_unique<Camera>();
        m_cameraPresets = std::make_unique<CameraPresets>();
        m_cameraPresets->CreateDefaultPresets();

        // Initialize UI state
        m_uiState.showHierarchyPanel = true;
        m_uiState.showPropertiesPanel = true;
        m_uiState.showViewControlsPanel = true;
        m_uiState.showVisualizationPanel = false;
        m_uiState.showResearchPanel = false;
        m_uiState.showPerformanceWindow = false;
        m_uiState.showStatusBar = true;
        m_uiState.showGrid = true;
        m_uiState.showDensityLayers = true;
        m_uiState.showObjectLabels = true;
        m_uiState.showConnections = false;
        m_uiState.densityLayerOpacity = 0.3f;
        m_uiState.gridOpacity = 0.5f;
        m_uiState.useCosmicColorPalette = true;
        m_uiState.statusBarHeight = 25.0f;

        // Initialize multiverse data
        m_multiverseData.currentHierarchyLevel = 0;
        m_multiverseData.selectedObjectId = "";

        // Initialize performance metrics
        m_performance.frameTime = 0.0f;
        m_performance.fps = 0.0f;
        m_performance.cameraUpdateTime = 0.0f;
        m_performance.uiRenderTime = 0.0f;
        m_performance.objectsRendered = 0;
    }

    Simulation::~Simulation()
    {
        if (m_deviceResources)
        {
            m_deviceResources->WaitForGpu();
        }
        CleanupImGui();
    }

    void Simulation::Initialize(HWND window, int width, int height)
    {
        m_windowHandle = window;

        m_deviceResources->SetWindow(window, width, height);
        m_deviceResources->CreateDeviceResources();
        CreateDeviceDependentResources();

        m_deviceResources->CreateWindowSizeDependentResources();
        CreateWindowSizeDependentResources();

        // Initialize camera with viewport size
        m_camera->SetViewportSize(width, height);

        // Initialize ImGui after device resources are ready
        InitializeImGui(window);

        // Create multiverse data
        CreateMultiverseHierarchy();

        // Set initial camera to multiverse overview
        m_cameraPresets->LoadPreset("multiverse", *m_camera);

        UpdateWindowTitle();
    }

    void Simulation::Tick()
    {
        m_timer.Tick([&]()
            {
                Update(m_timer);
            });

        Render();
    }

    void Simulation::Update(DX::StepTimer const& timer)
    {
        PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

        float deltaTime = float(timer.GetElapsedSeconds());

        // Update performance metrics
        m_performance.frameTime = deltaTime * 1000.0f;
        m_performance.fps = 1.0f / deltaTime;

        // Process input
        ProcessInput(deltaTime);

        // Update camera
        auto cameraStart = std::chrono::high_resolution_clock::now();
        m_camera->Update(deltaTime);
        auto cameraEnd = std::chrono::high_resolution_clock::now();
        m_performance.cameraUpdateTime = std::chrono::duration<float, std::milli>(cameraEnd - cameraStart).count();

        PIXEndEvent();
    }

    void Simulation::Render()
    {
        if (m_timer.GetFrameCount() == 0)
        {
            return;
        }

        // Start ImGui frame
        if (m_imguiInitialized)
        {
            ImGui_ImplDX12_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
        }

        // Prepare rendering
        m_deviceResources->Prepare();
        Clear();

        auto commandList = m_deviceResources->GetCommandList();
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render");

        // Render 3D scene
        RenderScene();

        // Render ImGui UI
        if (m_imguiInitialized)
        {
            auto uiStart = std::chrono::high_resolution_clock::now();
            RenderImGui();

            // Set descriptor heap for ImGui
            ID3D12DescriptorHeap* heaps[] = { m_imguiSrvHeap.Get() };
            commandList->SetDescriptorHeaps(_countof(heaps), heaps);

            // Render ImGui
            ImGui::Render();
            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);

            auto uiEnd = std::chrono::high_resolution_clock::now();
            m_performance.uiRenderTime = std::chrono::duration<float, std::milli>(uiEnd - uiStart).count();
        }

        PIXEndEvent(commandList);

        // Present frame
        PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");
        m_deviceResources->Present();
        m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());
        PIXEndEvent(m_deviceResources->GetCommandQueue());
    }

    void Simulation::RenderImGui()
    {
        // Apply cosmic dark theme
        ImGuiStyle& style = ImGui::GetStyle();
        if (m_uiState.useCosmicColorPalette) {
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.12f, 0.94f);
            style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.10f, 0.20f, 1.00f);
            style.Colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.40f, 0.55f);
            style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.25f, 0.50f, 0.80f);
            style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.30f, 0.30f, 0.60f, 1.00f);
            style.Colors[ImGuiCol_Button] = ImVec4(0.15f, 0.15f, 0.35f, 0.40f);
            style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.20f, 0.20f, 0.45f, 1.00f);
            style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.25f, 0.25f, 0.55f, 1.00f);
        }

        // Main menu bar
        RenderMainMenuBar();

        // Calculate available space for panels
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);

        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

        if (ImGui::Begin("DockSpace", nullptr, window_flags))
        {
            ImGui::PopStyleVar(3);

            // Create dockspace
            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

            // Render individual panels
            if (m_uiState.showHierarchyPanel) RenderHierarchyPanel();
            if (m_uiState.showPropertiesPanel) RenderPropertiesPanel();
            if (m_uiState.showViewControlsPanel) RenderViewControlsPanel();
            if (m_uiState.showVisualizationPanel) RenderVisualizationPanel();
            if (m_uiState.showResearchPanel) RenderResearchPanel();
            if (m_uiState.showPerformanceWindow) RenderPerformanceWindow();
        }
        ImGui::End();

        // Status bar
        if (m_uiState.showStatusBar) RenderStatusBar();
    }

    void Simulation::RenderMainMenuBar()
    {
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("Simulation"))
            {
                if (ImGui::MenuItem("Reset", "Ctrl+R"))
                {
                    ResetCamera();
                    CreateMultiverseHierarchy();
                }
                if (ImGui::MenuItem("Exit", "Alt+F4"))
                {
                    ExitSimulation();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("View"))
            {
                const char* currentMode = (m_camera->GetMode() == CameraMode::Orthographic2D) ? "2D (Orthographic)" : "3D (Perspective)";
                if (ImGui::MenuItem(currentMode, "Tab"))
                {
                    SwitchCameraMode();
                }

                ImGui::Separator();

                if (ImGui::BeginMenu("Camera Presets"))
                {
                    if (ImGui::MenuItem("Multiverse Overview"))
                        m_cameraPresets->LoadPreset("multiverse", *m_camera);
                    if (ImGui::MenuItem("Universe View"))
                        m_cameraPresets->LoadPreset("universe", *m_camera);
                    if (ImGui::MenuItem("Galaxy View"))
                        m_cameraPresets->LoadPreset("galaxy", *m_camera);
                    if (ImGui::MenuItem("Star System View"))
                        m_cameraPresets->LoadPreset("starsystem", *m_camera);
                    ImGui::EndMenu();
                }

                ImGui::Separator();
                ImGui::MenuItem("Hierarchy Panel", nullptr, &m_uiState.showHierarchyPanel);
                ImGui::MenuItem("Properties Panel", nullptr, &m_uiState.showPropertiesPanel);
                ImGui::MenuItem("View Controls", nullptr, &m_uiState.showViewControlsPanel);
                ImGui::MenuItem("Visualization", nullptr, &m_uiState.showVisualizationPanel);
                ImGui::MenuItem("Research Panel", nullptr, &m_uiState.showResearchPanel);
                ImGui::MenuItem("Performance", nullptr, &m_uiState.showPerformanceWindow);

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Navigate"))
            {
                if (ImGui::MenuItem("Go to Multiverse", "Home"))
                {
                    NavigateToHierarchyLevel(0);
                }
                if (ImGui::MenuItem("Go to Universe", "1"))
                {
                    NavigateToHierarchyLevel(1);
                }
                if (ImGui::MenuItem("Go to Galaxy", "2"))
                {
                    NavigateToHierarchyLevel(2);
                }
                if (ImGui::MenuItem("Go to Star System", "3"))
                {
                    NavigateToHierarchyLevel(3);
                }

                ImGui::Separator();
                if (ImGui::MenuItem("Reset Camera", "R"))
                {
                    ResetCamera();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Research"))
            {
                if (ImGui::MenuItem("Elena Danaan References"))
                {
                    m_uiState.showResearchPanel = true;
                }
                if (ImGui::MenuItem("Alex Collier References"))
                {
                    m_uiState.showResearchPanel = true;
                }
                if (ImGui::MenuItem("Dimensional Theory"))
                {
                    m_uiState.showResearchPanel = true;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Help"))
            {
                if (ImGui::MenuItem("Controls"))
                {
                    ShowControlsHelp();
                }
                if (ImGui::MenuItem("About Umgebung"))
                {
                    ShowAboutDialog();
                }
                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }
    }

    void Simulation::RenderHierarchyPanel()
    {
        if (ImGui::Begin("Multiverse Hierarchy", &m_uiState.showHierarchyPanel))
        {
            // Search/filter bar
            static std::string searchText;
            static char searchBuffer[256] = "";
            if (ImGui::InputText("Search", searchBuffer, sizeof(searchBuffer)))
            {
                searchText = searchBuffer;
            }

            ImGui::Separator();

            // Current location breadcrumb
            ImGui::Text("Current Location:");
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "%s", GetCurrentLocationString().c_str());

            ImGui::Separator();

            // Hierarchy tree
            if (ImGui::TreeNode("Multiverse (1)"))
            {
                for (int u = 1; u <= 3; ++u) // Show first 3 universes as example
                {
                    std::string universeLabel = "Universe " + std::to_string(u) + " of 10^100";
                    if (ImGui::TreeNode(universeLabel.c_str()))
                    {
                        for (int g = 1; g <= 3; ++g) // Show first 3 galaxies
                        {
                            std::string galaxyLabel = "Galaxy " + std::to_string(g) + " of 200B-3T";
                            if (ImGui::TreeNode(galaxyLabel.c_str()))
                            {
                                for (int s = 1; s <= 5; ++s) // Show first 5 stars
                                {
                                    std::string starLabel = "Star " + std::to_string(s) + " of 300B";
                                    if (ImGui::Selectable(starLabel.c_str()))
                                    {
                                        // Select this star
                                        std::string starId = "star_" + std::to_string(u) + "_" +
                                            std::to_string(g) + "_" + std::to_string(s);
                                        SelectObject(starId);
                                    }

                                    // Show density levels for selected star
                                    if (ImGui::IsItemHovered())
                                    {
                                        ImGui::BeginTooltip();
                                        ImGui::Text("Density Levels: 1-12");
                                        ImGui::Text("Current: Level 3 (Physical Matter)");
                                        ImGui::EndTooltip();
                                    }
                                }
                                ImGui::TreePop();
                            }
                        }
                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();
            }

            ImGui::Separator();

            // Quick navigation buttons
            if (ImGui::Button("Multiverse View"))
            {
                NavigateToHierarchyLevel(0);
            }
            ImGui::SameLine();
            if (ImGui::Button("Universe View"))
            {
                NavigateToHierarchyLevel(1);
            }

            if (ImGui::Button("Galaxy View"))
            {
                NavigateToHierarchyLevel(2);
            }
            ImGui::SameLine();
            if (ImGui::Button("Star System"))
            {
                NavigateToHierarchyLevel(3);
            }
        }
        ImGui::End();
    }

    void Simulation::RenderPropertiesPanel()
    {
        if (ImGui::Begin("Object Properties", &m_uiState.showPropertiesPanel))
        {
            auto selectedObject = GetSelectedObject();
            if (selectedObject)
            {
                ImGui::Text("Selected: %s", selectedObject->name.c_str());

                ImGui::Separator();

                ImGui::Text("Hierarchy Level: %d", selectedObject->hierarchyLevel);
                ImGui::Text("Density Level: %d of 12", selectedObject->densityLevel);

                ImGui::Separator();

                // Position
                ImGui::Text("Position:");
                ImGui::Text("  X: %.3f", selectedObject->position.x);
                ImGui::Text("  Y: %.3f", selectedObject->position.y);
                ImGui::Text("  Z: %.3f", selectedObject->position.z);

                // Physical properties
                ImGui::Separator();
                ImGui::Text("Physical Properties:");
                ImGui::Text("Radius: %.2f", selectedObject->radius);
                ImGui::Text("Mass: %.2e", selectedObject->mass);

                // Density information
                ImGui::Separator();
                ImGui::Text("Density Information:");
                ImGui::TextWrapped("Density represents the frequency rate of particles within this dimension. "
                    "Our dimension contains 12 density levels, with Source reached at the 13th level.");

                const char* densityNames[] = {
                    "Physical Matter", "Etheric", "Astral", "Mental Lower", "Mental Higher",
                    "Causal", "Buddhic", "Logoic", "Monadic", "Divine", "Source-1", "Source"
                };

                if (selectedObject->densityLevel >= 1 && selectedObject->densityLevel <= 12)
                {
                    ImGui::Text("Current Density: %s", densityNames[selectedObject->densityLevel - 1]);
                }

                // Research notes
                if (!selectedObject->researchNotes.empty())
                {
                    ImGui::Separator();
                    ImGui::Text("Research Notes:");
                    for (const auto& note : selectedObject->researchNotes)
                    {
                        ImGui::BulletText("%s", note.c_str());
                    }
                }
            }
            else
            {
                ImGui::Text("No object selected");
                ImGui::TextWrapped("Click on an object in the hierarchy or 3D view to see its properties.");
            }
        }
        ImGui::End();
    }

    void Simulation::RenderViewControlsPanel()
    {
        if (ImGui::Begin("View Controls", &m_uiState.showViewControlsPanel))
        {
            // Camera mode
            ImGui::Text("Camera Mode:");
            const char* currentMode = (m_camera->GetMode() == CameraMode::Orthographic2D) ? "2D (Orthographic)" : "3D (Perspective)";
            if (ImGui::Button(currentMode))
            {
                SwitchCameraMode();
            }

            ImGui::Separator();

            if (m_camera->GetMode() == CameraMode::Orthographic2D)
            {
                // 2D controls
                ImGui::Text("2D View Controls:");

                float zoom = m_camera->GetZoom2D();
                if (ImGui::SliderFloat("Zoom", &zoom, 0.001f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic))
                {
                    m_camera->SetZoom2D(zoom);
                }

                if (ImGui::Button("Reset 2D View"))
                {
                    m_camera->Reset2DView();
                }
            }
            else
            {
                // 3D controls
                ImGui::Text("3D View Controls:");

                float fov = m_camera->GetFieldOfView();
                if (ImGui::SliderFloat("Field of View", &fov, 1.0f, 120.0f))
                {
                    m_camera->SetFieldOfView(fov);
                }

                float speed = m_camera->GetMovementSpeed();
                if (ImGui::SliderFloat("Movement Speed", &speed, 0.1f, 50.0f))
                {
                    m_camera->SetMovementSpeed(speed);
                }

                float sensitivity = m_camera->GetMouseSensitivity();
                if (ImGui::SliderFloat("Mouse Sensitivity", &sensitivity, 0.01f, 1.0f))
                {
                    m_camera->SetMouseSensitivity(sensitivity);
                }

                if (ImGui::Button("Reset 3D View"))
                {
                    m_camera->Reset3DView();
                }
            }

            ImGui::Separator();

            // Camera presets
            ImGui::Text("Quick Views:");
            if (ImGui::Button("Multiverse", ImVec2(-1, 0)))
                m_cameraPresets->LoadPreset("multiverse", *m_camera);
            if (ImGui::Button("Universe", ImVec2(-1, 0)))
                m_cameraPresets->LoadPreset("universe", *m_camera);
            if (ImGui::Button("Galaxy", ImVec2(-1, 0)))
                m_cameraPresets->LoadPreset("galaxy", *m_camera);
            if (ImGui::Button("Star System", ImVec2(-1, 0)))
                m_cameraPresets->LoadPreset("starsystem", *m_camera);
        }
        ImGui::End();
    }

    void Simulation::RenderVisualizationPanel()
    {
        if (ImGui::Begin("Visualization Options", &m_uiState.showVisualizationPanel))
        {
            ImGui::Text("Display Options:");

            ImGui::Checkbox("Show Grid", &m_uiState.showGrid);
            ImGui::Checkbox("Show Density Layers", &m_uiState.showDensityLayers);
            ImGui::Checkbox("Show Object Labels", &m_uiState.showObjectLabels);
            ImGui::Checkbox("Show Connections", &m_uiState.showConnections);

            ImGui::Separator();

            ImGui::Text("Opacity Settings:");
            ImGui::SliderFloat("Density Layers", &m_uiState.densityLayerOpacity, 0.0f, 1.0f);
            ImGui::SliderFloat("Grid", &m_uiState.gridOpacity, 0.0f, 1.0f);

            ImGui::Separator();

            ImGui::Text("Color Scheme:");
            ImGui::Checkbox("Cosmic Palette", &m_uiState.useCosmicColorPalette);

            if (ImGui::CollapsingHeader("Density Level Colors"))
            {
                for (int i = 1; i <= 12; ++i)
                {
                    float hue = (i - 1) / 11.0f; // Rainbow spectrum
                    ImVec4 color = ImVec4(
                        0.5f + 0.5f * cosf(hue * 6.28f),
                        0.5f + 0.5f * cosf(hue * 6.28f + 2.09f),
                        0.5f + 0.5f * cosf(hue * 6.28f + 4.19f),
                        1.0f
                    );

                    ImGui::ColorButton(("Density " + std::to_string(i)).c_str(), color,
                        ImGuiColorEditFlags_NoTooltip, ImVec2(20, 20));
                    ImGui::SameLine();
                    ImGui::Text("Density Level %d", i);
                }
            }
        }
        ImGui::End();
    }

    void Simulation::RenderResearchPanel()
    {
        if (ImGui::Begin("Research References", &m_uiState.showResearchPanel))
        {
            if (ImGui::CollapsingHeader("Dimensional Theory", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::TextWrapped("According to Elena Danaan:");
                ImGui::BulletText("DIMENSION = Parallel universe");
                ImGui::BulletText("DENSITY = frequency rate of particles within a Dimension");
                ImGui::BulletText("Our dimension has 12 densities, with Source reached at 13th level");
                ImGui::BulletText("ONE TIMELINE - Time is like a train on one set of rails");
            }

            if (ImGui::CollapsingHeader("Multiverse Structure"))
            {
                ImGui::TextWrapped("Hierarchical Structure:");
                ImGui::BulletText("1 Multiverse");
                ImGui::BulletText("100^100 Universes per Multiverse");
                ImGui::BulletText("200 Billion - 3000 Trillion Galaxies per Universe");
                ImGui::BulletText("300 Billion Stars per Galaxy");
            }

            if (ImGui::CollapsingHeader("Key Researchers"))
            {
                if (ImGui::TreeNode("Elena Danaan"))
                {
                    ImGui::TextWrapped("Contact and dimensional researcher providing insights into "
                        "the structure of reality and density levels.");
                    if (ImGui::Button("Visit Website"))
                    {
                        // TODO: Open browser to https://www.elenadanaan.org/
                        ShellExecuteA(nullptr, "open", "https://www.elenadanaan.org/", nullptr, nullptr, SW_SHOWNORMAL);
                    }
                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Alex Collier"))
                {
                    ImGui::TextWrapped("Contactee sharing information about galactic civilizations "
                        "and cosmic structures.");
                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Nassim Haramein"))
                {
                    ImGui::TextWrapped("Physicist researching unified field theory and "
                        "the holographic nature of reality.");
                    ImGui::TreePop();
                }
            }

            if (ImGui::CollapsingHeader("The Flower of Life"))
            {
                ImGui::TextWrapped("\"The Flower of Life is the lattice structure of the Universe "
                    "and the key to the greatest power of all: the ability to affect "
                    "the hologram of reality.\" - Elena Danaan");
            }
        }
        ImGui::End();
    }

    void Simulation::RenderPerformanceWindow()
    {
        if (ImGui::Begin("Performance Metrics", &m_uiState.showPerformanceWindow))
        {
            ImGui::Text("Frame Time: %.3f ms (%.1f FPS)", m_performance.frameTime, m_performance.fps);
            ImGui::Text("Camera Update: %.3f ms", m_performance.cameraUpdateTime);
            ImGui::Text("UI Render: %.3f ms", m_performance.uiRenderTime);
            ImGui::Text("Objects Rendered: %d", m_performance.objectsRendered);

            // Simple performance graph
            static float frameTimeHistory[100] = {};
            static int frameTimeOffset = 0;

            frameTimeHistory[frameTimeOffset] = m_performance.frameTime;
            frameTimeOffset = (frameTimeOffset + 1) % 100;

            ImGui::PlotLines("Frame Time (ms)", frameTimeHistory, 100, frameTimeOffset,
                nullptr, 0.0f, 50.0f, ImVec2(0, 80));
        }
        ImGui::End();
    }

    void Simulation::RenderStatusBar()
    {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - m_uiState.statusBarHeight));
        ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, m_uiState.statusBarHeight));

        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;

        if (ImGui::Begin("StatusBar", nullptr, flags))
        {
            XMFLOAT3 pos = m_camera->GetPosition();
            ImGui::Text("Camera: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);

            ImGui::SameLine(200);
            const char* mode = (m_camera->GetMode() == CameraMode::Orthographic2D) ? "2D" : "3D";
            ImGui::Text("Mode: %s", mode);

            ImGui::SameLine(300);
            if (m_camera->GetMode() == CameraMode::Orthographic2D)
            {
                ImGui::Text("Zoom: %.3f", m_camera->GetZoom2D());
            }
            else
            {
                ImGui::Text("FOV: %.1f°", m_camera->GetFieldOfView());
            }

            ImGui::SameLine(450);
            ImGui::Text("Location: %s", GetCurrentLocationString().c_str());
        }
        ImGui::End();
    }

    void Simulation::RenderScene()
    {
        // TODO: Implement 3D scene rendering
        // This would render the actual multiverse objects, grids, density layers, etc.
        // For now, this is a placeholder that would use DirectX 12 to render geometry

        m_performance.objectsRendered = 42; // Placeholder value

        if (m_uiState.showGrid)
        {
            RenderGrid();
        }

        if (m_uiState.showDensityLayers)
        {
            RenderDensityLayers();
        }

        RenderMultiverseObjects();
    }

    void Simulation::RenderMultiverseObjects()
    {
        // TODO: Render multiverse objects based on current hierarchy level and camera view
        // This would involve:
        // 1. Culling objects outside view frustum
        // 2. Level-of-detail based on distance
        // 3. Rendering spheres/particles for different cosmic objects
        // 4. Applying density-based coloring and effects
    }

    void Simulation::RenderGrid()
    {
        // TODO: Render coordinate grid
        // In 2D mode: flat grid on XZ plane
        // In 3D mode: 3D coordinate system
    }

    void Simulation::RenderDensityLayers()
    {
        // TODO: Render density layers as colored transparent spheres/auras around objects
        // Each of the 12 density levels would have different colors and opacity
    }

    // Input handling methods
    void Simulation::ProcessInput(float deltaTime)
    {
        ProcessCameraInput(deltaTime);
        ProcessUIInput();
    }

    void Simulation::ProcessCameraInput(float deltaTime)
    {
        // Movement keys for 3D camera
        if (m_camera->GetMode() == CameraMode::Perspective3D)
        {
            if (m_keys['W'] || m_keys['w'])
                m_camera->ProcessKeyboard(CameraMovement::Forward, deltaTime);
            if (m_keys['S'] || m_keys['s'])
                m_camera->ProcessKeyboard(CameraMovement::Backward, deltaTime);
            if (m_keys['A'] || m_keys['a'])
                m_camera->ProcessKeyboard(CameraMovement::Left, deltaTime);
            if (m_keys['D'] || m_keys['d'])
                m_camera->ProcessKeyboard(CameraMovement::Right, deltaTime);
            if (m_keys['Q'] || m_keys['q'])
                m_camera->ProcessKeyboard(CameraMovement::Up, deltaTime);
            if (m_keys['E'] || m_keys['e'])
                m_camera->ProcessKeyboard(CameraMovement::Down, deltaTime);
        }

        // Speed modifiers
        float baseSpeed = 2.5f;
        if (m_keys[VK_SHIFT])
            baseSpeed *= 3.0f;
        if (m_keys[VK_CONTROL])
            baseSpeed *= 0.3f;

        m_camera->SetMovementSpeed(baseSpeed);
    }

    void Simulation::ProcessUIInput()
    {
        // Handle keyboard shortcuts that aren't handled by ImGui
        static bool tabPressed = false;
        if (m_keys[VK_TAB] && !tabPressed)
        {
            SwitchCameraMode();
            tabPressed = true;
        }
        else if (!m_keys[VK_TAB])
        {
            tabPressed = false;
        }

        // Reset camera
        static bool rPressed = false;
        if (m_keys['R'] && !rPressed)
        {
            ResetCamera();
            rPressed = true;
        }
        else if (!m_keys['R'])
        {
            rPressed = false;
        }

        // Navigation shortcuts
        static bool homePressed = false;
        if (m_keys[VK_HOME] && !homePressed)
        {
            NavigateToHierarchyLevel(0);
            homePressed = true;
        }
        else if (!m_keys[VK_HOME])
        {
            homePressed = false;
        }

        // Number keys for hierarchy levels
        static bool key1Pressed = false, key2Pressed = false, key3Pressed = false;
        if (m_keys['1'] && !key1Pressed)
        {
            NavigateToHierarchyLevel(1);
            key1Pressed = true;
        }
        else if (!m_keys['1'])
        {
            key1Pressed = false;
        }

        if (m_keys['2'] && !key2Pressed)
        {
            NavigateToHierarchyLevel(2);
            key2Pressed = true;
        }
        else if (!m_keys['2'])
        {
            key2Pressed = false;
        }

        if (m_keys['3'] && !key3Pressed)
        {
            NavigateToHierarchyLevel(3);
            key3Pressed = true;
        }
        else if (!m_keys['3'])
        {
            key3Pressed = false;
        }
    }

    // Camera control methods
    void Simulation::SwitchCameraMode()
    {
        m_camera->ToggleMode();
        UpdateWindowTitle();
    }

    void Simulation::ResetCamera()
    {
        m_camera->ResetToDefault();
    }

    void Simulation::FocusOnObject(const std::string& objectId)
    {
        auto it = m_multiverseData.objects.find(objectId);
        if (it != m_multiverseData.objects.end())
        {
            auto obj = it->second;
            m_camera->FocusOnObject(obj->position, obj->radius, 1.5f);
            SelectObject(objectId);
        }
    }

    // Object management methods
    void Simulation::CreateMultiverseHierarchy()
    {
        // Create sample multiverse data
        // This is a simplified version - in reality you'd load this from data files

        m_multiverseData.objects.clear();

        // Create multiverse root
        auto multiverse = std::make_shared<MultiverseObject>();
        multiverse->id = "multiverse_1";
        multiverse->name = "The Multiverse";
        multiverse->position = XMFLOAT3(0, 0, 0);
        multiverse->radius = 1000.0f;
        multiverse->mass = 1.0e50;
        multiverse->hierarchyLevel = 0;
        multiverse->densityLevel = 1;
        multiverse->color = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
        multiverse->description = "The complete multiverse containing 10^100 universes";
        multiverse->researchNotes.push_back("Root level of all existence");
        multiverse->citations.push_back("Elena Danaan - Dimensional Theory");
        m_multiverseData.objects[multiverse->id] = multiverse;

        // Create sample universes, galaxies, and stars
        for (int u = 1; u <= 3; ++u)
        {
            auto universe = std::make_shared<MultiverseObject>();
            universe->id = "universe_" + std::to_string(u);
            universe->name = "Universe " + std::to_string(u);
            universe->position = XMFLOAT3(u * 200.0f, 0, 0);
            universe->radius = 100.0f;
            universe->mass = 1.0e45;
            universe->hierarchyLevel = 1;
            universe->densityLevel = 1;
            universe->color = XMFLOAT4(0.8f, 0.8f, 1.0f, 1.0f);
            universe->parentId = "multiverse_1";
            universe->description = "A universe containing billions of galaxies";
            universe->researchNotes.push_back("One of 10^100 universes in the multiverse");
            m_multiverseData.objects[universe->id] = universe;

            for (int g = 1; g <= 3; ++g)
            {
                auto galaxy = std::make_shared<MultiverseObject>();
                galaxy->id = "galaxy_" + std::to_string(u) + "_" + std::to_string(g);
                galaxy->name = "Galaxy " + std::to_string(g);
                galaxy->position = XMFLOAT3(u * 200.0f + g * 30.0f, 0, g * 30.0f);
                galaxy->radius = 10.0f;
                galaxy->mass = 1.0e42;
                galaxy->hierarchyLevel = 2;
                galaxy->densityLevel = 1;
                galaxy->color = XMFLOAT4(0.8f, 1.0f, 0.8f, 1.0f);
                galaxy->parentId = universe->id;
                galaxy->description = "A spiral galaxy containing 300 billion stars";
                galaxy->researchNotes.push_back("Contains approximately 300 billion stars");
                m_multiverseData.objects[galaxy->id] = galaxy;

                for (int s = 1; s <= 5; ++s)
                {
                    auto star = std::make_shared<MultiverseObject>();
                    star->id = "star_" + std::to_string(u) + "_" + std::to_string(g) + "_" + std::to_string(s);
                    star->name = "Star " + std::to_string(s);
                    star->position = XMFLOAT3(
                        u * 200.0f + g * 30.0f + s * 3.0f,
                        0,
                        g * 30.0f + s * 2.0f
                    );
                    star->radius = 1.0f;
                    star->mass = 1.989e30f; // Solar mass
                    star->hierarchyLevel = 3;
                    star->densityLevel = 1 + (s % 12); // Cycle through density levels
                    star->color = XMFLOAT4(1.0f, 1.0f, 0.8f, 1.0f);
                    star->parentId = galaxy->id;
                    star->description = "A main sequence star with potential planetary system";
                    star->researchNotes.push_back("Physical matter density level");
                    star->citations.push_back("Elena Danaan - Dimensional Theory");
                    m_multiverseData.objects[star->id] = star;
                }
            }
        }
    }

    void Simulation::SelectObject(const std::string& objectId)
    {
        m_multiverseData.selectedObjectId = objectId;
    }

    std::shared_ptr<MultiverseObject> Simulation::GetSelectedObject() const
    {
        if (m_multiverseData.selectedObjectId.empty())
            return nullptr;

        auto it = m_multiverseData.objects.find(m_multiverseData.selectedObjectId);
        return (it != m_multiverseData.objects.end()) ? it->second : nullptr;
    }

    void Simulation::NavigateToHierarchyLevel(int level)
    {
        m_multiverseData.currentHierarchyLevel = level;

        switch (level)
        {
        case 0: // Multiverse
            m_cameraPresets->LoadPreset("multiverse", *m_camera);
            break;
        case 1: // Universe
            m_cameraPresets->LoadPreset("universe", *m_camera);
            break;
        case 2: // Galaxy
            m_cameraPresets->LoadPreset("galaxy", *m_camera);
            break;
        case 3: // Star system
            m_cameraPresets->LoadPreset("starsystem", *m_camera);
            break;
        }

        UpdateWindowTitle();
    }

    std::string Simulation::GetCurrentLocationString() const
    {
        switch (m_multiverseData.currentHierarchyLevel)
        {
        case 0: return "Multiverse Overview";
        case 1: return "Universe View";
        case 2: return "Galaxy View";
        case 3: return "Star System View";
        default: return "Unknown";
        }
    }

    void Simulation::UpdateWindowTitle()
    {
        if (m_windowHandle)
        {
            std::string title = "Umgebung - " + GetCurrentLocationString();
            title += " (" + std::string(m_camera->GetMode() == CameraMode::Orthographic2D ? "2D" : "3D") + ")";
            SetWindowTextA(m_windowHandle, title.c_str());
        }
    }

    void Simulation::ShowControlsHelp()
    {
        // This would show a modal dialog with controls information
        // For now, we'll use a simple message box
        std::string helpText =
            "Umgebung Controls:\n\n"
            "Camera Movement (3D Mode):\n"
            "  W/A/S/D - Move forward/left/backward/right\n"
            "  Q/E - Move up/down\n"
            "  Right mouse + drag - Look around\n"
            "  Mouse wheel - Zoom in/out\n"
            "  Shift - Move faster\n"
            "  Ctrl - Move slower\n\n"
            "Camera Movement (2D Mode):\n"
            "  Left mouse + drag - Pan view\n"
            "  Mouse wheel - Zoom in/out\n\n"
            "General Controls:\n"
            "  Tab - Switch between 2D/3D camera modes\n"
            "  R - Reset camera\n"
            "  Home - Go to Multiverse view\n"
            "  1/2/3 - Navigate to Universe/Galaxy/Star views\n"
            "  Ctrl+R - Reset simulation\n"
            "  Alt+F4 - Exit";

        MessageBoxA(m_windowHandle, helpText.c_str(), "Controls Help", MB_OK | MB_ICONINFORMATION);
    }

    void Simulation::ShowAboutDialog()
    {
        // This would show a modal dialog with about information
        std::string aboutText =
            "Umgebung - Multiverse Simulation\n\n"
            "A visualization tool for exploring the hierarchical structure\n"
            "of the multiverse based on dimensional theory research.\n\n"
            "Features:\n"
            "- Interactive 2D/3D visualization\n"
            "- Hierarchical navigation (Multiverse -> Universe -> Galaxy -> Stars)\n"
            "- 12 density level visualization\n"
            "- Research references and citations\n\n"
            "Based on research by:\n"
            "- Elena Danaan (Dimensional Theory)\n"
            "- Alex Collier (Galactic Structures)\n"
            "- Nassim Haramein (Unified Field Theory)\n\n"
            "Built with DirectX 12 and Dear ImGui";

        MessageBoxA(m_windowHandle, aboutText.c_str(), "About Umgebung", MB_OK | MB_ICONINFORMATION);
    }

    // Input event handlers
    void Simulation::OnMouseMove(float x, float y)
    {
        if (m_firstMouse)
        {
            m_lastMouseX = x;
            m_lastMouseY = y;
            m_firstMouse = false;
        }

        float xOffset = x - m_lastMouseX;
        float yOffset = m_lastMouseY - y; // Reversed since y-coordinates go top to bottom

        m_lastMouseX = x;
        m_lastMouseY = y;

        // Only process mouse movement for camera if right mouse button is held
        if (m_mouseButtons[1] && m_camera->GetMode() == CameraMode::Perspective3D)
        {
            m_camera->ProcessMouseMovement(xOffset, yOffset);
        }
        else if (m_mouseButtons[0] && m_camera->GetMode() == CameraMode::Orthographic2D)
        {
            // Pan in 2D mode with left mouse button
            m_camera->Pan2D(xOffset, yOffset);
        }
    }

    void Simulation::OnMouseButton(int button, bool pressed, float x, float y)
    {
        if (button >= 0 && button < 3)
        {
            m_mouseButtons[button] = pressed;

            // Handle object selection on left click release
            if (button == 0 && !pressed)
            {
                // TODO: Implement ray casting to select objects in 3D scene
                // For now, this is a placeholder
            }
        }
    }

    void Simulation::OnMouseWheel(float delta)
    {
        m_camera->ProcessMouseScroll(delta);
    }

    void Simulation::OnKeyPressed(int key)
    {
        if (key >= 0 && key < 256)
        {
            m_keys[key] = true;
        }
    }

    void Simulation::OnKeyReleased(int key)
    {
        if (key >= 0 && key < 256)
        {
            m_keys[key] = false;
        }
    }

    // Device resource methods
    void Simulation::InitializeImGui(HWND window)
    {
        // Create descriptor heap for ImGui
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = 1;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        if (FAILED(m_deviceResources->GetD3DDevice()->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&m_imguiSrvHeap))))
        {
            throw std::runtime_error("Failed to create ImGui descriptor heap");
        }

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer backends
        ImGui_ImplWin32_Init(window);
        ImGui_ImplDX12_Init(m_deviceResources->GetD3DDevice(),
            m_deviceResources->GetBackBufferCount(),
            m_deviceResources->GetBackBufferFormat(),
            m_imguiSrvHeap.Get(),
            m_imguiSrvHeap->GetCPUDescriptorHandleForHeapStart(),
            m_imguiSrvHeap->GetGPUDescriptorHandleForHeapStart());

        m_imguiInitialized = true;
    }

    void Simulation::CleanupImGui()
    {
        if (m_imguiInitialized)
        {
            ImGui_ImplDX12_Shutdown();
            ImGui_ImplWin32_Shutdown();
            ImGui::DestroyContext();
            m_imguiInitialized = false;
        }
    }

    void Simulation::Clear()
    {
        auto commandList = m_deviceResources->GetCommandList();
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Clear");

        // Clear the views
        const auto rtvDescriptor = m_deviceResources->GetRenderTargetView();
        const auto dsvDescriptor = m_deviceResources->GetDepthStencilView();

        commandList->OMSetRenderTargets(1, &rtvDescriptor, FALSE, &dsvDescriptor);

        // Use deep space color for background
        const float clearColor[] = { 0.02f, 0.02f, 0.08f, 1.0f }; // Deep space blue
        commandList->ClearRenderTargetView(rtvDescriptor, clearColor, 0, nullptr);
        commandList->ClearDepthStencilView(dsvDescriptor, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

        // Set the viewport and scissor rect
        const auto viewport = m_deviceResources->GetScreenViewport();
        const auto scissorRect = m_deviceResources->GetScissorRect();
        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);

        PIXEndEvent(commandList);
    }

    void Simulation::CreateDeviceDependentResources()
    {
        auto device = m_deviceResources->GetD3DDevice();

        // Check Shader Model 6 support
        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_0 };
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)))
            || (shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_0))
        {
#ifdef _DEBUG
            OutputDebugStringA("ERROR: Shader Model 6.0 is not supported!\n");
#endif
            throw std::runtime_error("Shader Model 6.0 is not supported!");
        }

        // Initialize DirectX Tool Kit
        m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

        // TODO: Initialize rendering resources for multiverse objects
        // This would include vertex buffers, shaders, textures, etc.
        // Example resources that would be created here:
        // - Sphere geometry for stars, galaxies, universes
        // - Grid line geometry
        // - Particle systems for density layers
        // - Shaders for different rendering modes
        // - Constant buffers for camera and lighting
    }

    void Simulation::CreateWindowSizeDependentResources()
    {
        // Update camera viewport
        auto outputSize = m_deviceResources->GetOutputSize();
        m_camera->SetViewportSize(outputSize.right, outputSize.bottom);

        // TODO: Update any render targets or resources that depend on window size
        // This might include:
        // - Shadow maps
        // - Post-processing render targets
        // - Screen-space effects buffers
    }

    void Simulation::OnDeviceLost()
    {
        // Cleanup ImGui
        CleanupImGui();

        // Cleanup DirectX Tool Kit
        m_graphicsMemory.reset();

        // TODO: Cleanup any other device-dependent resources
        // This would include releasing all D3D12 resources that need to be recreated
    }

    void Simulation::OnDeviceRestored()
    {
        CreateDeviceDependentResources();
        CreateWindowSizeDependentResources();

        // Reinitialize ImGui
        if (m_windowHandle)
        {
            InitializeImGui(m_windowHandle);
        }
    }

    // Window message handlers
    void Simulation::OnActivated()
    {
        // Resume any paused operations
        // This might include resuming animations, physics, or other time-based systems
    }

    void Simulation::OnDeactivated()
    {
        // Pause non-essential operations
        // This helps save resources when the window is not active
    }

    void Simulation::OnSuspending()
    {
        // Save state if needed
        // This is called when the application is being suspended (e.g., minimized)
    }

    void Simulation::OnResuming()
    {
        m_timer.ResetElapsedTime();
        // Restore state if needed
        // This is called when the application is being resumed
    }

    void Simulation::OnWindowMoved()
    {
        const auto r = m_deviceResources->GetOutputSize();
        m_deviceResources->WindowSizeChanged(r.right, r.bottom);
    }

    void Simulation::OnDisplayChange()
    {
        m_deviceResources->UpdateColorSpace();
    }

    void Simulation::OnWindowSizeChanged(int width, int height)
    {
        if (!m_deviceResources->WindowSizeChanged(width, height))
            return;

        CreateWindowSizeDependentResources();
        UpdateWindowTitle();
    }

    void Simulation::GetDefaultSize(int& width, int& height) const noexcept
    {
        width = 1280;
        height = 720;
    }

} // namespace Umgebung