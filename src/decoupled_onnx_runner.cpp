/**
 ******************************************************************************
 * @file           : src/decoupled_onnx_runner.cpp
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-20
 ******************************************************************************
 */

#include "decoupled_onnx_runner/decoupled_onxx_runner.h"
#include "log/logger.h"

DecoupledOnnxRunner::DecoupledOnnxRunner( unsigned int threads_num ) : threads_num_{ threads_num }
{
  INFO( logger, "DecoupledOnnxRunner created" );
}

DecoupledOnnxRunner::~DecoupledOnnxRunner()
{
  INFO( logger, "DecoupledOnnxRunner destroyed" );

  for ( auto& name : input_node_names_extractor_ )
  {
    delete[] name;
  }
  input_node_names_extractor_.clear();

  for ( auto& name : output_node_names_extractor_ )
  {
    delete[] name;
  }
  output_node_names_extractor_.clear();
}
