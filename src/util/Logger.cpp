/**
 * @file Logger.cpp
 * @brief Implements the Logger class.
 */
#include "umgebung/util/Logger.hpp"

namespace Umgebung::util {

    Logger& Logger::instance() {
        static Logger inst;
        return inst;
    }

    void Logger::init(const std::string& name,
        Level level,
        bool enableConsole,
        bool enableFile,
        bool enableConsolePanel,
        const std::string& filePath)
    {
        std::lock_guard<std::mutex> lock(m_initMutex);

        if (m_logger) return;

        std::vector<spdlog::sink_ptr> sinks;

        if (enableConsole) {
            auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console->set_pattern("%^[%T] %n: %v%$");
            sinks.push_back(console);
        }

        if (enableFile) {
            auto file = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filePath, true);
            file->set_pattern("[%T] [%l] %v");
            sinks.push_back(file);
        }

        if (enableConsolePanel) {
            m_panelSink = std::make_shared<ui::imgui::ImGuiConsoleSink_mt>();
            m_panelSink->set_pattern("[%T] [%^%l%$] %v"); // Example pattern with level
            sinks.push_back(m_panelSink);
        }

        m_logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());

        m_logger->set_level(static_cast<spdlog::level::level_enum>(level));

        spdlog::register_logger(m_logger);
    }

    const std::vector<std::string>& Logger::getPanelSinkBuffer() const {
        if (m_panelSink) {
            return m_panelSink->get_buffer();
        }
        static const std::vector<std::string> empty_buffer;
        return empty_buffer;
    }

    void Logger::clearPanelSinkBuffer() {
        if (m_panelSink) {
            m_panelSink->clear_buffer();
        }
    }

}