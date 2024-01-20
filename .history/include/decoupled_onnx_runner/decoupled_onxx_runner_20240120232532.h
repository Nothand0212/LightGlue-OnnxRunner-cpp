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

class DecoupledOnnxRunner : public BaseOnnxRunner
{
private:
  /* data */
public:
  DecoupledOnnxRunner( /* args */ );
  ~DecoupledOnnxRunner();
};
