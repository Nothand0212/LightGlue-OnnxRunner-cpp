#include <chrono>
#include <iostream>

class Timer
{
private:
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<double, std::milli>      time_consumed_ms;

public:
  double time_consumed_ms_double;
  Timer()
  {
  }

  ~Timer()
  {
  }

  void tic()
  {
    start = std::chrono::high_resolution_clock::now();
  }

  void toc()
  {
    end                     = std::chrono::high_resolution_clock::now();
    time_consumed_ms        = end - start;
    time_consumed_ms_double = time_consumed_ms.count();
  }
};