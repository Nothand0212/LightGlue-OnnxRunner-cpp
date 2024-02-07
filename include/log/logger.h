/**
 ******************************************************************************
 * @file           : include/combined_onnx_runner.cpp
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-18
 ******************************************************************************
 */

#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#define INFO  SPDLOG_LOGGER_INFO
#define WARN  SPDLOG_LOGGER_WARN
#define ERROR SPDLOG_LOGGER_ERROR
#define DEBUG SPDLOG_LOGGER_DEBUG
#define TRACE SPDLOG_LOGGER_TRACE

class Logger
{
private:
  spdlog::logger* logger_ptr;

  Logger() = default;

  Logger( Logger const& ) = delete;
  Logger& operator=( Logger const& ) = delete;

public:
  static Logger& getInstance()
  {
    static Logger instance;
    return instance;
  }

  spdlog::logger* getLogger()
  {
    return logger_ptr;
  }

  void initLogger( const std::string& log_path )
  {
    auto console_logger_sptr = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_logger_sptr    = std::make_shared<spdlog::sinks::basic_file_sink_mt>( log_path, true );
    logger_ptr               = new spdlog::logger( "MineLog", spdlog::sinks_init_list{ console_logger_sptr, file_logger_sptr } );

    // Set the log format
    logger_ptr->set_pattern( "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%@:%#] %v" );
  }
};
