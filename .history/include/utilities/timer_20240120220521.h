#pragma once
#include <chrono>
#include <iostream>

class Timer
{
private:
  std::chrono::high_resolution_clock::time_point start_, end_;
  std::chrono::duration<double, std::milli>      time_consumed_ms_;
  double                                         time_consumed_ms_double_;

public:
  Timer()  = default;
  ~Timer() = default;
  void tic()
  {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void toc()
  {
    end_                     = std::chrono::high_resolution_clock::now();
    time_consumed_ms_        = ( end_ - start_ ).count();
    time_consumed_ms_double_ = time_consumed_ms_.count();
  }

  double getDuration() const
  {
    return time_consumed_ms_double_;
  }
};