#pragma once

#include <spdlog/sinks/base_sink.h>
#include <vector>
#include <string>
#include <mutex>

namespace Umgebung::ui::imgui {

    // A custom spdlog sink that writes to an in-memory buffer for ImGui
    template<typename Mutex>
    class ImGuiConsoleSink : public spdlog::sinks::base_sink<Mutex> {
    public:
        // Provides thread-safe access to the log buffer
        const std::vector<std::string>& get_buffer() const {
            return m_buffer;
        }

        void clear_buffer() {
            std::lock_guard<Mutex> lock(this->mutex_);
            m_buffer.clear();
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& msg) override {
            // Format the message
            spdlog::memory_buf_t formatted;
            this->formatter_->format(msg, formatted);

            // Store it in our buffer
            m_buffer.push_back(fmt::to_string(formatted));
        }

        void flush_() override {} // No action needed on flush

    private:
        std::vector<std::string> m_buffer;
    };

    // Use a standard mutex for thread safety
    using ImGuiConsoleSink_mt = ImGuiConsoleSink<std::mutex>;

} // namespace Umgebung::ui::imgui