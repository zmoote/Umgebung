#pragma once

#include "../walbourn/DeviceResources.h"
#include "../walbourn/StepTimer.h"
#include "../umgebung/Camera.h"

#include <memory>
#include <unordered_map>
#include <string>

// ImGui includes
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx12.h>

namespace Umgebung {

    // Forward declarations
    struct MultiverseObject;
    enum class UIPanel;

    class Simulation final : public DX::IDeviceNotify
    {
    public:
        Simulation() noexcept(false);
        ~Simulation();

        Simulation(Simulation&&) = default;
        Simulation& operator= (Simulation&&) = default;

        Simulation(Simulation const&) = delete;
        Simulation& operator= (Simulation const&) = delete;

        // Initialization and management
        void Initialize(HWND window, int width, int height);

        // Basic simulation loop
        void Tick();

        // IDeviceNotify
        void OnDeviceLost() override;
        void OnDeviceRestored() override;

        // Messages
        void OnActivated();
        void OnDeactivated();
        void OnSuspending();
        void OnResuming();
        void OnWindowMoved();
        void OnDisplayChange();
        void OnWindowSizeChanged(int width, int height);

        // Input handling
        void OnMouseMove(float x, float y);
        void OnMouseButton(int button, bool pressed, float x, float y);
        void OnMouseWheel(float delta);
        void OnKeyPressed(int key);
        void OnKeyReleased(int key);

        // Camera control
        void SwitchCameraMode();
        void ResetCamera();
        void FocusOnObject(const std::string& objectId);

        // Properties
        void GetDefaultSize(int& width, int& height) const noexcept;

    private:
        // Core update and render
        void Update(DX::StepTimer const& timer);
        void Render();
        void Clear();

        // UI Rendering
        void RenderImGui();
        void RenderMainMenuBar();
        void RenderHierarchyPanel();
        void RenderPropertiesPanel();
        void RenderViewControlsPanel();
        void RenderVisualizationPanel();
        void RenderResearchPanel();
        void RenderStatusBar();
        void RenderPerformanceWindow();

        void ShowControlsHelp();
        void ShowAboutDialog();

        // 3D Scene rendering
        void RenderScene();
        void RenderMultiverseObjects();
        void RenderGrid();
        void RenderDensityLayers();

        // Input processing
        void ProcessInput(float deltaTime);
        void ProcessCameraInput(float deltaTime);
        void ProcessUIInput();

        // Object management
        void CreateMultiverseHierarchy();
        void SelectObject(const std::string& objectId);
        std::shared_ptr<MultiverseObject> GetSelectedObject() const;
        void NavigateToHierarchyLevel(int level);

        // Device resources
        void CreateDeviceDependentResources();
        void CreateWindowSizeDependentResources();

        // ImGui setup and cleanup
        void InitializeImGui(HWND window);
        void CleanupImGui();

        // Utility functions
        std::string GetCurrentLocationString() const;
        void UpdateWindowTitle();

    private:
        // Core systems
        std::unique_ptr<DX::DeviceResources> m_deviceResources;
        DX::StepTimer m_timer;
        std::unique_ptr<DirectX::GraphicsMemory> m_graphicsMemory;

        // Camera system
        std::unique_ptr<Camera> m_camera;
        std::unique_ptr<CameraPresets> m_cameraPresets;

        // ImGui resources
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_imguiSrvHeap;
        bool m_imguiInitialized;

        // Window and input state
        HWND m_windowHandle;
        bool m_keys[256];
        bool m_mouseButtons[3];
        float m_lastMouseX, m_lastMouseY;
        bool m_firstMouse;
        bool m_mouseCaptured;

        // UI State
        struct UIState {
            bool showHierarchyPanel = true;
            bool showPropertiesPanel = true;
            bool showViewControlsPanel = true;
            bool showVisualizationPanel = true;
            bool showResearchPanel = false;
            bool showPerformanceWindow = true;
            bool showStatusBar = true;

            float hierarchyPanelWidth = 300.0f;
            float propertiesPanelWidth = 350.0f;
            float statusBarHeight = 25.0f;

            // View options
            bool showGrid = true;
            bool showDensityLayers = true;
            bool showObjectLabels = true;
            bool showConnections = true;

            // Visualization settings
            float densityLayerOpacity = 0.3f;
            float gridOpacity = 0.5f;
            bool useCosmicColorPalette = true;
        } m_uiState;

        // Multiverse data
        struct MultiverseData {
            std::unordered_map<std::string, std::shared_ptr<MultiverseObject>> objects;
            std::string selectedObjectId;
            int currentHierarchyLevel = 0; // 0=Multiverse, 1=Universe, 2=Galaxy, 3=Star
            std::vector<std::string> navigationHistory;
            int historyIndex = -1;
        } m_multiverseData;

        // Performance tracking
        struct PerformanceMetrics {
            float frameTime = 0.0f;
            float fps = 0.0f;
            size_t memoryUsage = 0;
            int objectsRendered = 0;
            float cameraUpdateTime = 0.0f;
            float uiRenderTime = 0.0f;
        } m_performance;
    };

    // Multiverse object structure
    struct MultiverseObject {
        std::string id;
        std::string name;
        DirectX::XMFLOAT3 position;
        float radius;
        int hierarchyLevel; // 0=Multiverse, 1=Universe, 2=Galaxy, 3=Star
        int densityLevel;   // 1-12 density levels

        // Hierarchy relationships
        std::string parentId;
        std::vector<std::string> childIds;

        // Visualization properties
        DirectX::XMFLOAT4 color;
        bool visible;
        float opacity;

        // Research data
        std::string description;
        std::vector<std::string> researchNotes;
        std::vector<std::string> citations;

        // Physics properties (for future simulation)
        DirectX::XMFLOAT3 velocity;
        float mass;

        MultiverseObject() :
            position(0.0f, 0.0f, 0.0f),
            radius(1.0f),
            hierarchyLevel(0),
            densityLevel(1),
            color(1.0f, 1.0f, 1.0f, 1.0f),
            visible(true),
            opacity(1.0f),
            velocity(0.0f, 0.0f, 0.0f),
            mass(1.0f) {
        }
    };

    // UI Panel enumeration for state management
    enum class UIPanel {
        Hierarchy,
        Properties,
        ViewControls,
        Visualization,
        Research,
        Performance,
        StatusBar
    };

}