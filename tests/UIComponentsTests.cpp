/**
 * @file UIComponentsTests.cpp
 * @brief Automated CTest verification suite for GUI-exclusive `umgebung::ui`
 *        components (`ImGuiConsoleSink` thread-safe ring buffer, `Panel` state
 *        machine, and `SimulationPlaybackState` string helpers).
 *
 * @details
 * Executes headless without creating an OS window, Vulkan device, or ImGui
 * context so CI and automated verification runs validate UI data structures
 * cleanly on desktop builds (`UMGEBUNG_BUILD_GUI=ON`).
 */

#include "umgebung/ui/ImGuiConsoleSink.hpp"
#include "umgebung/ui/Panel.hpp"
#include "umgebung/ui/UIContext.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <spdlog/logger.h>

namespace {

    /** @brief Minimal concrete `Panel` subclass for testing lifecycle & visibility methods. */
    class DummyPanel final : public umgebung::ui::Panel {
    public:
        explicit DummyPanel(std::string name, bool isOpen = true)
            : Panel(std::move(name), isOpen) {}

        void OnUIRender(umgebung::ui::UIContext& /*ctx*/) override {
            ++renderCount;
        }

        int renderCount{0};
    };

    bool TestImGuiConsoleSinkRingBuffer() {
        auto sink = std::make_shared<umgebung::ui::ImGuiConsoleSink>(3U);
        sink->set_pattern("[%l] %v");

        spdlog::logger testLogger("ui_test_logger", sink);
        testLogger.set_level(spdlog::level::trace);

        testLogger.info("First message");
        testLogger.warn("Second message");
        testLogger.error("Third message");
        testLogger.debug("Fourth message"); // Should evict "First message"

        if (sink->GetEntryCount() != 3U) {
            std::cerr << "Expected 3 entries in bounded ring buffer, got " << sink->GetEntryCount() << std::endl;
            return false;
        }

        std::vector<umgebung::ui::ConsoleLogEntry> entries;
        sink->CopyEntries(entries);
        if (entries.size() != 3U) {
            std::cerr << "CopyEntries returned unexpected size: " << entries.size() << std::endl;
            return false;
        }

        if (entries[0].level != spdlog::level::warn || entries[0].message.find("Second message") == std::string::npos) {
            std::cerr << "Expected FIFO eviction of oldest message; got: " << entries[0].message << std::endl;
            return false;
        }
        if (entries[2].level != spdlog::level::debug || entries[2].message.find("Fourth message") == std::string::npos) {
            std::cerr << "Expected newest message at tail; got: " << entries[2].message << std::endl;
            return false;
        }
        if (!entries[2].message.empty() && entries[2].message.back() == '\n') {
            std::cerr << "Expected trailing newline to be trimmed by ImGuiConsoleSink." << std::endl;
            return false;
        }

        sink->Clear();
        if (sink->GetEntryCount() != 0U) {
            std::cerr << "Expected Clear() to empty ImGuiConsoleSink." << std::endl;
            return false;
        }
        return true;
    }

    bool TestPanelLifecycleAndPlaybackEnum() {
        DummyPanel panel("Hierarchy", true);
        if (panel.GetName() != "Hierarchy" || !panel.IsOpen() || panel.IsFocused() || panel.IsHovered()) {
            std::cerr << "Unexpected initial Panel state." << std::endl;
            return false;
        }

        panel.Close();
        if (panel.IsOpen()) {
            std::cerr << "Panel::Close() failed." << std::endl;
            return false;
        }

        panel.Toggle();
        if (!panel.IsOpen()) {
            std::cerr << "Panel::Toggle() failed." << std::endl;
            return false;
        }

        if (std::string(umgebung::ui::SimulationPlaybackStateToString(umgebung::ui::SimulationPlaybackState::Playing)) != "Playing" ||
            std::string(umgebung::ui::SimulationPlaybackStateToString(umgebung::ui::SimulationPlaybackState::Paused)) != "Paused" ||
            std::string(umgebung::ui::SimulationPlaybackStateToString(umgebung::ui::SimulationPlaybackState::Stopped)) != "Stopped") {
            std::cerr << "SimulationPlaybackStateToString mismatch." << std::endl;
            return false;
        }
        return true;
    }

} // anonymous namespace

int main() {
    if (!TestImGuiConsoleSinkRingBuffer()) {
        return EXIT_FAILURE;
    }
    if (!TestPanelLifecycleAndPlaybackEnum()) {
        return EXIT_FAILURE;
    }
    std::cout << "All UIComponentsTests passed!" << std::endl;
    return EXIT_SUCCESS;
}
