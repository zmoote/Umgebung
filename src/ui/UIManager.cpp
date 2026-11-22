/**
 * @file UIManager.cpp
 * @brief Implementation of `umgebung::ui::UIManager` delivering the canonical
 *        4-region docked workstation layout, Main Menu Bar, `ImGuiConsoleSink`
 *        log bridge, modal `FilePickerPanel`, and modular panel orchestration
 *        adapted from all branches of `Umgebung-legacy`.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/UIManager.hpp"
#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/ui/panels/AboutPanel.hpp"
#include "umgebung/ui/panels/ConsolePanel.hpp"
#include "umgebung/ui/panels/HierarchyPanel.hpp"
#include "umgebung/ui/panels/PropertiesPanel.hpp"
#include "umgebung/ui/panels/StatisticsPanel.hpp"
#include "umgebung/ui/panels/ViewportPanel.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include <imgui.h>
#include <imgui_internal.h>
#include <spdlog/spdlog.h>

namespace umgebung::ui {

    UIManager::UIManager() = default;

    UIManager::~UIManager() {
        Shutdown();
    }

    void UIManager::Initialize() {
        if (m_initialized) {
            return;
        }

        // 1. Attach thread-safe ImGuiConsoleSink to spdlog's default logger so
        //    all engine diagnostics stream live into the bottom Console panel.
        m_consoleSink = std::make_shared<ImGuiConsoleSink>();
        m_consoleSink->set_pattern("[%T] [%l] %v");
        if (auto defaultLogger = spdlog::default_logger()) {
            defaultLogger->sinks().push_back(m_consoleSink);
        }

        // 2. Instantiate the canonical legacy modular panel set:
        //    - Default docked workstation panels (open on startup):
        //        Viewport (Center), Hierarchy (Left 20%), Properties (Right 25%), Console (Bottom 25%).
        //    - On-demand floating utility panels (closed until opened via Menu Bar):
        //        Statistics (Tools -> Statistics), About (Help -> About Umgebung).
        m_panels.clear();
        m_panels.push_back(std::make_unique<ViewportPanel>());
        m_panels.push_back(std::make_unique<HierarchyPanel>());
        m_panels.push_back(std::make_unique<PropertiesPanel>());
        m_panels.push_back(std::make_unique<ConsolePanel>(m_consoleSink));
        m_panels.push_back(std::make_unique<StatisticsPanel>());
        m_panels.push_back(std::make_unique<AboutPanel>());

        m_filePickerPanel = std::make_unique<FilePickerPanel>();

        m_layoutInitialized = false;
        m_forceResetLayout = false;
        m_initialized = true;

        spdlog::info("UIManager initialized (4-pane workstation layout: Hierarchy | Viewport | Properties / Console).");
    }

    void UIManager::Shutdown() noexcept {
        if (!m_initialized && m_panels.empty() && m_consoleSink == nullptr) {
            return;
        }

        // Detach our console sink cleanly before destroying it so late shutdown
        // spdlog messages never dereference a torn-down UI sink.
        if (m_consoleSink != nullptr) {
            if (auto defaultLogger = spdlog::default_logger()) {
                auto& sinks = defaultLogger->sinks();
                sinks.erase(
                    std::remove(sinks.begin(), sinks.end(), m_consoleSink),
                    sinks.end()
                );
            }
            m_consoleSink.reset();
        }

        m_panels.clear();
        m_filePickerPanel.reset();
        m_initialized = false;
        m_layoutInitialized = false;
        m_forceResetLayout = false;
    }

    void UIManager::OpenFilePicker(
        const std::string& title,
        const std::string& buttonLabel,
        FilePickerPanel::FileSelectedCallback callback,
        const std::vector<std::string>& extensions,
        const std::filesystem::path& startPath,
        const std::string& defaultName
    ) {
        if (m_filePickerPanel != nullptr) {
            m_filePickerPanel->OpenModal(
                title,
                buttonLabel,
                std::move(callback),
                extensions,
                startPath,
                defaultName
            );
        }
    }

    std::filesystem::path UIManager::ResolveDefaultScenesDirectory(const UIContext& ctx) const {
        std::error_code ec;
        if (!ctx.currentScenePath.empty() && ctx.currentScenePath.has_parent_path()) {
            const auto parent = ctx.currentScenePath.parent_path();
            if (std::filesystem::is_directory(parent, ec)) {
                return parent;
            }
        }
        const std::filesystem::path localScenes("scenes");
        if (std::filesystem::is_directory(localScenes, ec)) {
            return localScenes;
        }
        return std::filesystem::current_path(ec);
    }

    void UIManager::SetupDockspace() {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const ImGuiID dockspaceId = ImGui::GetID("MainDockSpace");
        const ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;

        // Query BEFORE `DockSpaceOverViewport()` creates an empty root node:
        //   - Returns `nullptr` on first run (no `imgui.ini`) OR when an older
        //     `imgui.ini` from a pre-refactor layout has no `"MainDockSpace"` node.
        //   - Returns non-null on subsequent runs once the user's docked layout
        //     has been saved to `imgui.ini`, preserving their customizations.
        const bool missingSavedDockLayout =
            (!m_layoutInitialized && (ImGui::DockBuilderGetNode(dockspaceId) == nullptr));

        ImGui::DockSpaceOverViewport(dockspaceId, viewport, dockspaceFlags);

        // Build the canonical legacy 4-region workstation layout when no saved
        // `"MainDockSpace"` exists or when explicitly requested via
        // `View -> Reset Default Layout`.
        const bool shouldBuildLayout = m_forceResetLayout || missingSavedDockLayout;

        if (shouldBuildLayout) {
            m_layoutInitialized = true;
            m_forceResetLayout = false;

            ImGui::DockBuilderRemoveNode(dockspaceId);
            ImGui::DockBuilderAddNode(dockspaceId, dockspaceFlags | ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

            // Exact legacy split geometry:
            //   - Left   20% : Hierarchy
            //   - Right  25% : Properties
            //   - Bottom 25% : Console (under Viewport)
            //   - Center     : Viewport
            ImGuiID dockMainId = dockspaceId;
            const ImGuiID dockLeftId = ImGui::DockBuilderSplitNode(dockMainId, ImGuiDir_Left, 0.20f, nullptr, &dockMainId);
            const ImGuiID dockRightId = ImGui::DockBuilderSplitNode(dockMainId, ImGuiDir_Right, 0.25f, nullptr, &dockMainId);
            const ImGuiID dockBottomId = ImGui::DockBuilderSplitNode(dockMainId, ImGuiDir_Down, 0.25f, nullptr, &dockMainId);

            ImGui::DockBuilderDockWindow("Hierarchy", dockLeftId);
            ImGui::DockBuilderDockWindow("Properties", dockRightId);
            ImGui::DockBuilderDockWindow("Console", dockBottomId);
            ImGui::DockBuilderDockWindow("Viewport", dockMainId);
            ImGui::DockBuilderFinish(dockspaceId);

            // Re-open all four primary docked panels when layout is built/reset.
            if (auto* p = GetPanel<HierarchyPanel>())  { p->Open(); }
            if (auto* p = GetPanel<PropertiesPanel>()) { p->Open(); }
            if (auto* p = GetPanel<ConsolePanel>())    { p->Open(); }
            if (auto* p = GetPanel<ViewportPanel>())   { p->Open(); }
        } else {
            m_layoutInitialized = true;
        }
    }

    void UIManager::RenderMainMenuBar(UIContext& ctx) {
        if (!ImGui::BeginMainMenuBar()) {
            return;
        }

        // ---------------------------------------------------------------------
        // 1. File Menu (New / Save / Save As / Open / Exit — Legacy Structure)
        // ---------------------------------------------------------------------
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Scene")) {
                if (ctx.onNewScene) {
                    ctx.onNewScene();
                }
            }
            ImGui::Separator();

            const bool canSave = (ctx.hasActiveScene && ctx.activeScene != nullptr);
            if (ImGui::MenuItem("Save Scene", nullptr, false, canSave)) {
                if (!ctx.currentScenePath.empty() && ctx.onSaveSceneFile) {
                    (void)ctx.onSaveSceneFile(ctx.currentScenePath);
                } else {
                    const std::string defaultFile =
                        (ctx.activeScene != nullptr && !ctx.activeScene->universeId.empty())
                            ? (ctx.activeScene->universeId + ".scenario.umgebung")
                            : std::string("untitled.scenario.umgebung");
                    OpenFilePicker(
                        "Save Scene As...",
                        "Save",
                        [onSave = ctx.onSaveSceneFile](const std::filesystem::path& path) {
                            if (onSave) { (void)onSave(path); }
                        },
                        {".umgebung"},
                        ResolveDefaultScenesDirectory(ctx),
                        defaultFile
                    );
                }
            }

            if (ImGui::MenuItem("Save As...", nullptr, false, canSave)) {
                const std::string defaultFile =
                    (ctx.activeScene != nullptr && !ctx.activeScene->universeId.empty())
                        ? (ctx.activeScene->universeId + ".scenario.umgebung")
                        : std::string("untitled.scenario.umgebung");
                OpenFilePicker(
                    "Save Scene As...",
                    "Save",
                    [onSave = ctx.onSaveSceneFile](const std::filesystem::path& path) {
                        if (onSave) { (void)onSave(path); }
                    },
                    {".umgebung"},
                    ResolveDefaultScenesDirectory(ctx),
                    defaultFile
                );
            }

            if (ImGui::MenuItem("Open Scene...")) {
                OpenFilePicker(
                    "Open Scene",
                    "Open",
                    [onOpen = ctx.onOpenSceneFile](const std::filesystem::path& path) {
                        if (onOpen) { (void)onOpen(path); }
                    },
                    {".umgebung"},
                    ResolveDefaultScenesDirectory(ctx)
                );
            }

            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                if (ctx.onRequestExit) {
                    ctx.onRequestExit();
                }
            }
            ImGui::EndMenu();
        }

        // ---------------------------------------------------------------------
        // 2. Simulation Menu (Play / Pause / Stop / Step / Reset — Legacy Structure)
        // ---------------------------------------------------------------------
        if (ImGui::BeginMenu("Simulation")) {
            const bool isPlayable =
                ctx.hasActiveScene && ctx.activeScene != nullptr &&
                ctx.activeScene->domain != scene::SimulationDomain::RadialAndScaleProfile;

            if (ImGui::MenuItem("Play", nullptr, ctx.playbackState == SimulationPlaybackState::Playing, isPlayable)) {
                if (ctx.onPlay) { ctx.onPlay(); }
            }
            if (ImGui::MenuItem("Pause", nullptr, ctx.playbackState == SimulationPlaybackState::Paused, isPlayable)) {
                if (ctx.onPause) { ctx.onPause(); }
            }
            if (ImGui::MenuItem("Stop", nullptr, ctx.playbackState == SimulationPlaybackState::Stopped, isPlayable)) {
                if (ctx.onStop) { ctx.onStop(); }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Single Step", nullptr, false, isPlayable)) {
                if (ctx.onSingleStep) { ctx.onSingleStep(); }
            }
            if (ImGui::MenuItem("Reset Scene", nullptr, false, ctx.hasActiveScene)) {
                if (ctx.onResetSimulation) { ctx.onResetSimulation(); }
            }
            ImGui::EndMenu();
        }

        // ---------------------------------------------------------------------
        // 3. Modules Menu (Legacy dev "Modules" adapted to Domain-Filtered ModelRegistry)
        // ---------------------------------------------------------------------
        if (ImGui::BeginMenu("Modules")) {
            if (!ctx.hasActiveScene || ctx.activeScene == nullptr) {
                ImGui::TextDisabled("No active scene loaded.");
            } else {
                const auto compatible = models::ModelRegistry::GetModelsForDomain(ctx.activeScene->domain);
                ImGui::TextDisabled("Primary Model A (%s):", scene::SimulationDomainToString(ctx.activeScene->domain));
                ImGui::Separator();
                for (const auto& desc : compatible) {
                    const bool isSelected = (desc.id == ctx.primaryModelId);
                    if (ImGui::MenuItem(desc.displayName, desc.tierLabel, isSelected)) {
                        if (!isSelected && ctx.onSelectPrimaryModel) {
                            ctx.onSelectPrimaryModel(desc.id);
                        }
                    }
                }

                ImGui::Separator();
                bool overlayEnabled = ctx.compareOverlayEnabled;
                if (ImGui::MenuItem("Simultaneous A/B Overlay", nullptr, &overlayEnabled)) {
                    if (ctx.onSetCompareOverlay) {
                        ctx.onSetCompareOverlay(overlayEnabled, ctx.compareModelId);
                    }
                }
                if (ImGui::BeginMenu("Overlay Model B", ctx.compareOverlayEnabled)) {
                    for (const auto& desc : compatible) {
                        const bool isSelected = (desc.id == ctx.compareModelId);
                        if (ImGui::MenuItem(desc.displayName, desc.tierLabel, isSelected)) {
                            if (!isSelected && ctx.onSetCompareOverlay) {
                                ctx.onSetCompareOverlay(true, desc.id);
                            }
                        }
                    }
                    ImGui::EndMenu();
                }
            }
            ImGui::EndMenu();
        }

        // ---------------------------------------------------------------------
        // 4. Tools Menu (Statistics / Console — Legacy Structure)
        // ---------------------------------------------------------------------
        if (ImGui::BeginMenu("Tools")) {
            if (auto* stats = GetPanel<StatisticsPanel>()) {
                ImGui::MenuItem("Statistics", nullptr, stats->GetOpenPtr());
            }
            if (auto* console = GetPanel<ConsolePanel>()) {
                ImGui::MenuItem("Console", nullptr, console->GetOpenPtr());
            }
            ImGui::EndMenu();
        }

        // ---------------------------------------------------------------------
        // 5. View Menu (Mode Toggles + Panel Visibility + Reset Default Layout)
        // ---------------------------------------------------------------------
        if (ImGui::BeginMenu("View")) {
            if (ctx.viewportAutoFit != nullptr &&
                ctx.hasActiveScene && ctx.activeScene != nullptr &&
                ctx.activeScene->domain == scene::SimulationDomain::NBodyDynamics) {
                ImGui::MenuItem("Auto-Fit Orbital View", nullptr, ctx.viewportAutoFit);
            }
            if (ctx.showFlowerOfLifeLatticeView != nullptr &&
                ctx.flowerMultiverse != nullptr && !ctx.flowerMultiverse->IsEmpty()) {
                ImGui::MenuItem("Flower of Life Lattice", nullptr, ctx.showFlowerOfLifeLatticeView);
            }
            ImGui::Separator();

            if (auto* p = GetPanel<HierarchyPanel>())  { ImGui::MenuItem("Hierarchy",  nullptr, p->GetOpenPtr()); }
            if (auto* p = GetPanel<PropertiesPanel>()) { ImGui::MenuItem("Properties", nullptr, p->GetOpenPtr()); }
            if (auto* p = GetPanel<ViewportPanel>())   { ImGui::MenuItem("Viewport",   nullptr, p->GetOpenPtr()); }
            if (auto* p = GetPanel<ConsolePanel>())    { ImGui::MenuItem("Console",    nullptr, p->GetOpenPtr()); }

            ImGui::Separator();
            if (ImGui::MenuItem("Reset Default Layout")) {
                RequestResetLayout();
            }
            ImGui::EndMenu();
        }

        // ---------------------------------------------------------------------
        // 6. Help Menu (About Umgebung — Legacy Structure)
        // ---------------------------------------------------------------------
        if (ImGui::BeginMenu("Help")) {
            if (auto* about = GetPanel<AboutPanel>()) {
                if (ImGui::MenuItem("About Umgebung")) {
                    about->Open();
                }
            }
            ImGui::EndMenu();
        }

        // ---------------------------------------------------------------------
        // Compact Live Engine Telemetry Strip on Menu Bar
        // ---------------------------------------------------------------------
        const ImGuiIO& io = ImGui::GetIO();
        ImGui::Separator();
        ImGui::TextDisabled(
            " %.1f FPS | %s | %s | %s%s",
            io.Framerate,
            scene::SimulationScopeToString(ctx.activeScope),
            (ctx.hasActiveScene && ctx.activeScene != nullptr)
                ? scene::SimulationDomainToString(ctx.activeScene->domain)
                : "No Scene",
            ctx.hasActiveScene ? scene::ModelIdToString(ctx.primaryModelId) : "-",
            ctx.compareOverlayEnabled ? " [A/B]" : ""
        );

        ImGui::EndMainMenuBar();
    }

    void UIManager::Render(UIContext& ctx) {
        if (!m_initialized) {
            return;
        }

        // 1. Establish the full-viewport dockspace & default 4-region layout.
        SetupDockspace();

        // 2. Render the top Main Menu Bar.
        RenderMainMenuBar(ctx);

        // 3. Render every modular docked & floating panel.
        for (auto& panel : m_panels) {
            if (panel != nullptr) {
                panel->OnUIRender(ctx);
            }
        }

        // 4. Render the modal file picker popup whenever active.
        if (m_filePickerPanel != nullptr) {
            m_filePickerPanel->OnUIRender(ctx);
        }
    }

} // namespace umgebung::ui
