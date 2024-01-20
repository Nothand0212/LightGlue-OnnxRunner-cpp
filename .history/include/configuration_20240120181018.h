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
#include <iostream>

#include "nlohmann/json.hpp"
#include "string"
class Config
{
public:
  std::string matcher_path   = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint_lightglue_fused.onnx";  // light_glue
  std::string extractor_path = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint.onnx";                  // only super point
  std::string combiner_path  = "/home/lin/CLionProjects/light_glue_onnx/models/superpoint_2048_lightglue_end2end.onnx";

  std::string image_src_path = "/home/lin/Projects/vision_ws/data/left";
  std::string image_dst_path = "/home/lin/Projects/vision_ws/data/right";

  std::string output_path = "/home/lin/CLionProjects/light_glue_onnx/output";

  bool gray_flag = true;

  unsigned int image_size = 2048;
  float        threshold  = 0.05f;

  std::string device{ "cuda" };

public:
  Config()  = default;
  ~Config() = default;

  void readConfig( const std::string& config_path )
  {
    std::ifstream  config_file( config_file_path );
    nlohmann::json j;
    i >> j;

    matcher_path   = j[ "matcher_path" ];
    extractor_path = j[ "extractor_path" ];
    combiner_path  = j[ "combiner_path" ];

    image_src_path = j[ "image_src_path" ];
    image_dst_path = j[ "image_dst_path" ];

    output_path = j[ "output_path" ];

    gray_flag = j[ "gray_flag" ];

    image_size = j[ "image_size" ];
    threshold  = j[ "threshold" ];

    device = j[ "device" ];
  }
};