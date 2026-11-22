/**
 * @file ImGuiConsoleSink.hpp
 * @brief Thread-safe custom `spdlog` sink streaming engine diagnostics directly
 *        into the docked Dear ImGui `ConsolePanel`.
 *
 * @details
 * Adapted directly from the `ImGuiConsoleSink` pattern used across the legacy
 * Umgebung branches (`main`, `dev`, `opengl-backend`, `vulkan-backend`), with
 * three modern hardening upgrades:
 *   1. Stores structured `ConsoleLogEntry` records preserving each message's
 *      `spdlog::level::level_enum` so `ConsolePanel` can color-code warnings,
 *      errors, and informational lines without fragile substring parsing.
 *   2. Enforces a fixed-capacity ring buffer (`kDefaultMaxEntries = 2048`) so
 *      long-running overnight simulations never experience unbounded heap growth.
 *   3. Provides a mutex-protected `CopyEntries()` snapshot helper so the UI
 *      render thread never holds the logger lock while issuing ImGui draw calls.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/ImGuiConsoleSink.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/sinks/base_sink.h>

/**
 * @namespace umgebung::ui
 * @brief Desktop-only Dear ImGui workstation layout manager (`UIManager`),
 *        logging sink, shared `UIContext`, and modular docked/floating panels.
 */
namespace umgebung::ui {

    /**
     * @struct ConsoleLogEntry
     * @brief Single formatted log line captured by `ImGuiConsoleSink` for display
     *        inside `ConsolePanel`.
     */
    struct ConsoleLogEntry {
        /** @brief Severity level emitted by `spdlog` (used for level filtering and tinting). */
        spdlog::level::level_enum level{spdlog::level::info};

        /** @brief Pre-formatted UTF-8 log string produced by the sink's `spdlog` formatter. */
        std::string message;
    };

    /**
     * @class ImGuiConsoleSink
     * @brief Custom thread-safe `spdlog` sink backed by a bounded ring buffer.
     *
     * @details
     * Inherits from `spdlog::sinks::base_sink<std::mutex>`, which acquires
     * `base_sink::mutex_` around every `sink_it_()` and `flush_()` invocation.
     * Public inspection methods (`CopyEntries()`, `Clear()`, `GetEntryCount()`)
     * acquire the exact same mutex to guarantee data-race-free access across
     * worker threads and the GUI render thread.
     */
    class ImGuiConsoleSink final : public spdlog::sinks::base_sink<std::mutex> {
    public:
        /** @brief Default maximum number of retained log lines before oldest entries roll off. */
        static constexpr std::size_t kDefaultMaxEntries = 2048U;

        /**
         * @brief Constructs an `ImGuiConsoleSink` with a configurable ring capacity.
         *
         * @param maxEntries Maximum number of log lines retained in memory
         *                   (clamped to at least `1` to avoid empty-buffer traps).
         */
        explicit ImGuiConsoleSink(std::size_t maxEntries = kDefaultMaxEntries) noexcept
            : m_maxEntries(maxEntries > 0U ? maxEntries : 1U) {}

        /**
         * @brief Atomically copies all currently buffered log entries into `outEntries`.
         *
         * @param outEntries Destination vector overwritten with the current log history.
         */
        void CopyEntries(std::vector<ConsoleLogEntry>& outEntries) const {
            std::lock_guard<std::mutex> lock(m_entriesMutex);
            outEntries.assign(m_entries.begin(), m_entries.end());
        }

        /**
         * @brief Clears all retained log entries from the console ring buffer.
         */
        void Clear() {
            std::lock_guard<std::mutex> lock(m_entriesMutex);
            m_entries.clear();
        }

        /**
         * @brief Returns the number of log entries currently buffered.
         *
         * @return Current count of retained `ConsoleLogEntry` items.
         */
        [[nodiscard]] std::size_t GetEntryCount() const {
            std::lock_guard<std::mutex> lock(m_entriesMutex);
            return m_entries.size();
        }

        /**
         * @brief Returns the configured ring-buffer capacity.
         *
         * @return Maximum number of retained entries before FIFO eviction.
         */
        [[nodiscard]] std::size_t GetMaxEntries() const noexcept {
            return m_maxEntries;
        }

    protected:
        /**
         * @brief Formats and appends an incoming `spdlog` message under the sink mutex.
         *
         * @param msg Incoming structured log record from `spdlog`.
         */
        void sink_it_(const spdlog::details::log_msg& msg) override {
            spdlog::memory_buf_t formatted;
            this->formatter_->format(msg, formatted);

            // Trim trailing newline characters so ImGui::TextUnformatted does not
            // render extra blank vertical padding beneath every console line.
            std::string line(formatted.data(), formatted.size());
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                line.pop_back();
            }

            std::lock_guard<std::mutex> lock(m_entriesMutex);
            if (m_entries.size() >= m_maxEntries) {
                m_entries.pop_front();
            }
            m_entries.push_back(ConsoleLogEntry{msg.level, std::move(line)});
        }

        /** @brief No-op flush hook (entries are stored immediately in memory). */
        void flush_() override {}

    private:
        /** @brief Maximum ring-buffer capacity before evicting the oldest entry. */
        std::size_t m_maxEntries{kDefaultMaxEntries};

        /** @brief Dedicated `mutable` mutex guarding `m_entries` for const-correct reads. */
        mutable std::mutex m_entriesMutex;

        /** @brief FIFO ring buffer of captured log entries guarded by `m_entriesMutex`. */
        std::deque<ConsoleLogEntry> m_entries;
    };

} // namespace umgebung::ui
