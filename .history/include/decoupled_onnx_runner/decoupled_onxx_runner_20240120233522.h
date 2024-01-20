/**
 ******************************************************************************
 * @file           : include/decoupled_onnx_runner/decoupled_onxx_runner.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-20
 ******************************************************************************
 */

#pragma once

#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>

#include "base_onnx_runner.h"
#include "configuration.h"
#include "image_process.h"
#include "utilities/accumulate_average.h"
#include "utilities/timer.h"

class DecoupledOnnxRunner : public BaseOnnxRunner
{
private:
  unsigned int threads_num_;

  Ort::Env                      env_extractor_, env_matcher_;
  Ort::SessionOptions           session_options_extractor_, session_options_matcher_;
  std::unique_ptr<Ort::Session> session_uptr_extractor_, session_uptr_matcher_;

  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<char*>                input_node_names_extractor_;
  std::vector<std::vector<int64_t>> input_node_shapes_extractor_;
  std::vector<char*>                output_node_names_extractor_;
  std::vector<std::vector<int64_t>> output_node_shapes_extractor_;


  std::vector<char*>                input_node_names_matcher_;
  std::vector<std::vector<int64_t>> input_node_shapes_matcher_;
  std::vector<char*>                output_node_names_matcher_;
  std::vector<std::vector<int64_t>> output_node_shapes_matcher_;

  float match_threshold_{ 0.0f };
  Timer timer_extractor_, timer_matcher_;


public:
  DecoupledOnnxRunner( /* args */ );
  ~DecoupledOnnxRunner();
};
