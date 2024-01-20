/**
 ******************************************************************************
 * @file           : configuration.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-17
 ******************************************************************************
 */

#pragma once
#include "string"
struct Config
{
  std::string matcher_path   = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint_lightglue_fused.onnx";  // light_glue
  std::string extractor_path = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint.onnx";                  // only super point
  std::string combiner_path  = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint_2048_lightglue_end2end.onnx";

  std::string image_src_path = "/home/lin/Projects/vision_ws/data/left";
  std::string image_dst_path = "/home/lin/Projects/vision_ws/data/right";

  std::string output_path = "/home/lin/CLionProjects/light_glue_onnx/output";

  bool gray_flag = true;

  unsigned int image_size = 1024;
  float        threshold  = 0.05f;

  std::string device{ "cuda" };
};