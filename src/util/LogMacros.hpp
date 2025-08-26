#pragma once
#include "Logger.hpp"

#define UMGEBUNG_LOG_TRACE(...) Umgebung::util::Logger::instance().trace(__VA_ARGS__)
#define UMGEBUNG_LOG_DEBUG(...) Umgebung::util::Logger::instance().debug(__VA_ARGS__)
#define UMGEBUNG_LOG_INFO(...)  Umgebung::util::Logger::instance().info(__VA_ARGS__)
#define UMGEBUNG_LOG_WARN(...)  Umgebung::util::Logger::instance().warn(__VA_ARGS__)
#define UMGEBUNG_LOG_ERROR(...) Umgebung::util::Logger::instance().error(__VA_ARGS__)
#define UMGEBUNG_LOG_CRIT(...)  Umgebung::util::Logger::instance().critical(__VA_ARGS__)