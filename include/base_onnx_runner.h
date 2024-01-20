/**
 ******************************************************************************
 * @file           : base_onnx_runner.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */

#pragma once

#include "configuration.h"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/core/mat.hpp"
#include "opencv4/opencv2/core/types.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "vector"

class BaseOnnxRunner
{
public:
  virtual int initOrtEnv( const Config& config )
  {
    return EXIT_SUCCESS;
  }

  virtual float getMatchThreshold()
  {
    return 0.0f;
  }

  virtual void setMatchThreshold( const float& threshold ) {}

  virtual double getTimer( const std::string& name )
  {
    return 0.0f;
  }

  virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
  inferenceImage( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst )
  {
    return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>{};
  }

  virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
  getKeyPointsResult()
  {
    return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>{};
  }
};