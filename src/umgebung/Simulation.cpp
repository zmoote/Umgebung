#include "../../include/walbourn/pch.h"
#include "../../include/umgebung/Simulation.h"

extern void ExitSimulation() noexcept;

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace Umgebung {

    Simulation::Simulation() noexcept(false)
        : m_imguiInitialized(false)
    {
        m_deviceResources = std::make_unique<DX::DeviceResources>();
        // TODO: Provide parameters for swapchain format, depth/stencil format, and backbuffer count.
        //   Add DX::DeviceResources::c_AllowTearing to opt-in to variable rate displays.
        //   Add DX::DeviceResources::c_EnableHDR for HDR10 display.
        //   Add DX::DeviceResources::c_ReverseDepth to optimize depth buffer clears for 0 instead of 1.
        m_deviceResources->RegisterDeviceNotify(this);
    }

    Simulation::~Simulation()
    {
        if (m_deviceResources)
        {
            m_deviceResources->WaitForGpu();
        }

        CleanupImGui();
    }

    // Initialize the Direct3D resources required to run.
    void Simulation::Initialize(HWND window, int width, int height)
    {
        m_deviceResources->SetWindow(window, width, height);

        m_deviceResources->CreateDeviceResources();
        CreateDeviceDependentResources();

        m_deviceResources->CreateWindowSizeDependentResources();
        CreateWindowSizeDependentResources();

        // Initialize ImGui after device resources are ready
        InitializeImGui(window);

        // TODO: Change the timer settings if you want something other than the default variable timestep mode.
        // e.g. for 60 FPS fixed timestep update logic, call:
        /*
        m_timer.SetFixedTimeStep(true);
        m_timer.SetTargetElapsedSeconds(1.0 / 60);
        */
    }

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
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        // ImGui::StyleColorsLight();

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

#pragma region Frame Update
    // Executes the basic game loop.
    void Simulation::Tick()
    {
        m_timer.Tick([&]()
            {
                Update(m_timer);
            });

        Render();
    }

    // Updates the world.
    void Simulation::Update(DX::StepTimer const& timer)
    {
        PIXBeginEvent(PIX_COLOR_DEFAULT, L"Update");

        float elapsedTime = float(timer.GetElapsedSeconds());

        // TODO: Add your simulation logic here.
        elapsedTime;

        PIXEndEvent();
    }
#pragma endregion

#pragma region Frame Render
    // Draws the scene.
    void Simulation::Render()
    {
        // Don't try to render anything before the first Update.
        if (m_timer.GetFrameCount() == 0)
        {
            return;
        }

        // Start the Dear ImGui frame
        if (m_imguiInitialized)
        {
            ImGui_ImplDX12_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
        }

        // Prepare the command list to render a new frame.
        m_deviceResources->Prepare();
        Clear();

        auto commandList = m_deviceResources->GetCommandList();
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Render");

        // TODO: Add your rendering code here.

        // Render ImGui
        if (m_imguiInitialized)
        {
            RenderImGui();

            // Set descriptor heap for ImGui
            ID3D12DescriptorHeap* heaps[] = { m_imguiSrvHeap.Get() };
            commandList->SetDescriptorHeaps(_countof(heaps), heaps);

            // Render ImGui
            ImGui::Render();
            ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), commandList);
        }

        PIXEndEvent(commandList);

        // Show the new frame.
        PIXBeginEvent(m_deviceResources->GetCommandQueue(), PIX_COLOR_DEFAULT, L"Present");
        m_deviceResources->Present();

        // If using the DirectX Tool Kit for DX12, uncomment this line:
        m_graphicsMemory->Commit(m_deviceResources->GetCommandQueue());

        PIXEndEvent(m_deviceResources->GetCommandQueue());
    }

    void Simulation::RenderImGui()
    {
        // Example ImGui windows for your UFO/Multiverse simulation

        // Main menu bar
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("Simulation"))
            {
                if (ImGui::MenuItem("Reset", "Ctrl+R"))
                {
                    // TODO: Reset simulation
                }
                if (ImGui::MenuItem("Exit", "Alt+F4"))
                {
                    ExitSimulation();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View"))
            {
                if (ImGui::MenuItem("2D View", "2"))
                {
                    // TODO: Switch to 2D multiverse hierarchy view
                }
                if (ImGui::MenuItem("3D View", "3"))
                {
                    // TODO: Switch to 3D exploration view
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("About"))
            {
                if (ImGui::MenuItem("About Umgebung"))
                {
                    // TODO: Show about dialog
                }
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        // Simulation Control Panel
        ImGui::Begin("Simulation Controls");
        {
            ImGui::Text("Umgebung - Reality Simulation");
            ImGui::Separator();

            static bool show_2d_view = true;
            static bool show_3d_view = false;
            static float simulation_speed = 1.0f;

            ImGui::Checkbox("2D Multiverse View", &show_2d_view);
            ImGui::Checkbox("3D Exploration View", &show_3d_view);

            ImGui::SliderFloat("Simulation Speed", &simulation_speed, 0.1f, 10.0f);

            if (ImGui::Button("Reset Simulation"))
            {
                // TODO: Implement simulation reset
            }

            ImGui::Text("Frame Time: %.3f ms (%.1f FPS)",
                m_timer.GetElapsedSeconds() * 1000.0f,
                1.0f / m_timer.GetElapsedSeconds());
        }
        ImGui::End();

        // Cosmic Hierarchy Explorer (based on your UFO disclosure concepts)
        ImGui::Begin("Cosmic Hierarchy");
        {
            ImGui::Text("Navigate the Structure of Reality");
            ImGui::Separator();

            if (ImGui::TreeNode("Multiverse"))
            {
                if (ImGui::TreeNode("Our Universe"))
                {
                    if (ImGui::TreeNode("Local Group"))
                    {
                        if (ImGui::TreeNode("Milky Way Galaxy"))
                        {
                            if (ImGui::TreeNode("Solar System"))
                            {
                                if (ImGui::Selectable("Earth"))
                                {
                                    // TODO: Focus on Earth
                                }
                                if (ImGui::Selectable("Mars"))
                                {
                                    // TODO: Focus on Mars
                                }
                                if (ImGui::Selectable("Ceres (Secret Colony?)"))
                                {
                                    // Reference to Tony Rodrigues' accounts
                                }
                                ImGui::TreePop();
                            }
                            ImGui::TreePop();
                        }
                        ImGui::TreePop();
                    }
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("Parallel Universes"))
                {
                    ImGui::Text("Based on UFO Disclosure Community insights");
                    // TODO: Add parallel universe exploration
                    ImGui::TreePop();
                }
                ImGui::TreePop();
            }
        }
        ImGui::End();

        // Entity Information Panel (for ETs, consciousness levels, etc.)
        ImGui::Begin("Entity Database");
        {
            ImGui::Text("Consciousness and Beings");
            ImGui::Separator();

            static int selected_category = 0;
            const char* categories[] = { "Galactic Federation", "Negative ETs", "Earth Beings", "Consciousness Levels" };
            ImGui::Combo("Category", &selected_category, categories, IM_ARRAYSIZE(categories));

            switch (selected_category)
            {
            case 0: // Galactic Federation
                ImGui::Text("Based on Elena Danaan's contacts:");
                ImGui::BulletText("Pleiadians");
                ImGui::BulletText("Arcturians");
                ImGui::BulletText("Andromedans");
                break;
            case 1: // Negative ETs
                ImGui::Text("Based on disclosure accounts:");
                ImGui::BulletText("Grays (various factions)");
                ImGui::BulletText("Reptilians");
                break;
            case 2: // Earth Beings
                ImGui::Text("Terrestrial consciousness:");
                ImGui::BulletText("Humans");
                ImGui::BulletText("Underground civilizations");
                break;
            case 3: // Consciousness Levels
                ImGui::Text("Dimensional awareness levels:");
                ImGui::BulletText("3D Physical");
                ImGui::BulletText("4D Astral");
                ImGui::BulletText("5D+ Higher Dimensions");
                break;
            }
        }
        ImGui::End();

        // Physics Parameters (Nassim Haramein concepts)
        ImGui::Begin("Quantum Physics Parameters");
        {
            ImGui::Text("Unified Field Theory Parameters");
            ImGui::Separator();

            static float holographic_mass = 1.0f;
            static float vacuum_energy = 1.0f;
            static float consciousness_field = 1.0f;

            ImGui::SliderFloat("Holographic Mass Density", &holographic_mass, 0.1f, 2.0f);
            ImGui::SliderFloat("Vacuum Energy Level", &vacuum_energy, 0.1f, 2.0f);
            ImGui::SliderFloat("Consciousness Field Strength", &consciousness_field, 0.1f, 2.0f);

            ImGui::Text("These parameters affect the simulation's");
            ImGui::Text("representation of reality's fundamental structure.");
        }
        ImGui::End();
    }

    // Helper method to clear the back buffers.
    void Simulation::Clear()
    {
        auto commandList = m_deviceResources->GetCommandList();
        PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Clear");

        // Clear the views.
        const auto rtvDescriptor = m_deviceResources->GetRenderTargetView();
        const auto dsvDescriptor = m_deviceResources->GetDepthStencilView();

        commandList->OMSetRenderTargets(1, &rtvDescriptor, FALSE, &dsvDescriptor);
        commandList->ClearRenderTargetView(rtvDescriptor, Colors::CornflowerBlue, 0, nullptr);
        commandList->ClearDepthStencilView(dsvDescriptor, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

        // Set the viewport and scissor rect.
        const auto viewport = m_deviceResources->GetScreenViewport();
        const auto scissorRect = m_deviceResources->GetScissorRect();
        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);

        PIXEndEvent(commandList);
    }
#pragma endregion

#pragma region Message Handlers
    // Message handlers
    void Simulation::OnActivated()
    {
        // TODO: Simulation is becoming active window.
    }

    void Simulation::OnDeactivated()
    {
        // TODO: Simulation is becoming background window.
    }

    void Simulation::OnSuspending()
    {
        // TODO: Simulation is being power-suspended (or minimized).
    }

    void Simulation::OnResuming()
    {
        m_timer.ResetElapsedTime();

        // TODO: Simulation is being power-resumed (or returning from minimize).
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

        // TODO: Simulation window is being resized.
    }

    // Properties
    void Simulation::GetDefaultSize(int& width, int& height) const noexcept
    {
        // TODO: Change to desired default window size (note minimum size is 320x200).
        width = 1280;
        height = 720;
    }
#pragma endregion

#pragma region Direct3D Resources
    // These are the resources that depend on the device.
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

        // If using the DirectX Tool Kit for DX12, uncomment this line:
        m_graphicsMemory = std::make_unique<GraphicsMemory>(device);

        // TODO: Initialize device dependent objects here (independent of window size).
    }

    // Allocate all memory resources that change on a window SizeChanged event.
    void Simulation::CreateWindowSizeDependentResources()
    {
        // TODO: Initialize windows-size dependent objects here.
    }

    void Simulation::OnDeviceLost()
    {
        // TODO: Add Direct3D resource cleanup here.

        // Cleanup ImGui
        CleanupImGui();

        // If using the DirectX Tool Kit for DX12, uncomment this line:
        m_graphicsMemory.reset();
    }

    void Simulation::OnDeviceRestored()
    {
        CreateDeviceDependentResources();
        CreateWindowSizeDependentResources();

        // Reinitialize ImGui after device restoration
        // Note: We need the window handle, which should be stored or obtained from device resources
        // For now, this is a TODO - you may need to store the window handle in the class
        // InitializeImGui(m_windowHandle);
    }
#pragma endregion
}