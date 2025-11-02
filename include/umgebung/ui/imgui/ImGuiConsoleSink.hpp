#pragma once

#include <spdlog/sinks/base_sink.h>
#include <vector>
#include <string>
#include <mutex>

namespace Umgebung::ui::imgui {

    /**
 * @file ImGuiConsoleSink.hpp
 * @brief Contains the ImGuiConsoleSink class.
 */
#pragma once

#include <spdlog/sinks/base_sink.h>
#include <vector>
#include <string>
#include <mutex>

namespace Umgebung::ui::imgui {

    /**
     * @brief A custom spdlog sink that writes to an in-memory buffer for ImGui.
     * 
     * @tparam Mutex The mutex type to use.
     */
    template<typename Mutex>
    class ImGuiConsoleSink : public spdlog::sinks::base_sink<Mutex> {
    public:
        /**
         * @brief Provides thread-safe access to the log buffer.
         * 
         * @return const std::vector<std::string>& 
         */
        const std::vector<std::string>& get_buffer() const {
            return m_buffer;
        }

        /**
         * @brief Clears the log buffer.
         */
        void clear_buffer() {
            std::lock_guard<Mutex> lock(this->mutex_);
            m_buffer.clear();
        }

    protected:
        /**
         * @brief Sinks the log message.
         * 
         * @param msg The log message.
         */
        void sink_it_(const spdlog::details::log_msg& msg) override {
            spdlog::memory_buf_t formatted;
            this->formatter_->format(msg, formatted);

            m_buffer.push_back(fmt::to_string(formatted));
        }

        /**
         * @brief Flushes the log buffer.
         */
        void flush_() override {} 

    private:
        std::vector<std::string> m_buffer; ///< The log buffer.
    };

    /**
     * @brief A thread-safe ImGui console sink.
     */
    using ImGuiConsoleSink_mt = ImGuiConsoleSink<std::mutex>;

} // namespace Umgebung::ui::imgui

} // namespace Umgebung::ui::imgui