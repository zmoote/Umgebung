#pragma once
#include "Logger.hpp"

/**
 * @file LogMacros.hpp
 * @brief Contains macros for logging.
 */
#pragma once
#include "Logger.hpp"

/**
 * @brief Logs a trace message.
 */
#define UMGEBUNG_LOG_TRACE(...) Umgebung::util::Logger::instance().trace(__VA_ARGS__)

/**
 * @brief Logs a debug message.
 */
#define UMGEBUNG_LOG_DEBUG(...) Umgebung::util::Logger::instance().debug(__VA_ARGS__)

/**
 * @brief Logs an info message.
 */
#define UMGEBUNG_LOG_INFO(...)  Umgebung::util::Logger::instance().info(__VA_ARGS__)

/**
 * @brief Logs a warning message.
 */
#define UMGEBUNG_LOG_WARN(...)  Umgebung::util::Logger::instance().warn(__VA_ARGS__)

/**
 * @brief Logs an error message.
 */
#define UMGEBUNG_LOG_ERROR(...) Umgebung::util::Logger::instance().error(__VA_ARGS__)

/**
 * @brief Logs a critical message.
 */
#define UMGEBUNG_LOG_CRIT(...)  Umgebung::util::Logger::instance().critical(__VA_ARGS__)