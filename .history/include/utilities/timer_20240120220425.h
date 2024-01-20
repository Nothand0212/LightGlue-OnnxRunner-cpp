#pragma once
#include <chrono>
#include <iostream>

class Timer
{
private:
  std::chrono::high_resolution_clock::time_point start_, end_;
  std::chrono::duration<double, std::milli>      time_consumed_ms_;

public:
  double time_consumed_ms_double_;
  Timer()
  {
  }

  ~Timer()
  {
  }

  void tic()
  {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void toc()
  {
    end_                     = std::chrono::high_resolution_clock::now();
    time_consumed_ms_        = end_ - start_;
    time_consumed_ms_double_ = time_consumed_ms_.count();
  }

  double getDuration()
  {
    return time_consumed_ms_double_;
  }
};