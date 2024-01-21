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

int DecoupledOnnxRunner::initOrtEnv( const Config& config )
{
  INFO( logger, "initializing Ort Env" );

  try
  {
    env_extractor_ = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "DecoupledOnnxRunner Extractor" );
    env_matcher_   = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "DecoupledOnnxRunner Matcher" );

    // create session options
    session_options_extractor_ = Ort::SessionOptions();
    session_options_matcher_   = Ort::SessionOptions();

    if ( threads_num_ == 0 )
    {
      threads_num_ = std::thread::hardware_concurrency();
    }

    session_options_extractor_.SetIntraOpNumThreads( threads_num_ );
    session_options_extractor_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    session_options_matcher_.SetIntraOpNumThreads( threads_num_ );
    session_options_matcher_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    INFO( logger, "Using {0} threads, with graph optimization level {1}", threads_num_, GraphOptimizationLevel::ORT_ENABLE_ALL );

    if ( config.device == "cuda" )
    {
      INFO( logger, "Using CUDA provider with default options" );

      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id                 = 0;                              // 这行设置 CUDA 设备 ID 为 0，这意味着 ONNX Runtime 将在第一个 CUDA 设备（通常是第一个 GPU）上运行模型。
      cuda_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchDefault;  // 这行设置 cuDNN 卷积算法搜索策略为默认值。cuDNN 是 NVIDIA 的深度神经网络库，它包含了许多用于卷积的优化算法。
      cuda_options.gpu_mem_limit             = 0;                              // 这行设置 GPU 内存限制为 0，这意味着 ONNX Runtime 可以使用所有可用的 GPU 内存。
      cuda_options.arena_extend_strategy     = 1;                              // 这行设置内存分配策略为 1，这通常意味着 ONNX Runtime 将在需要更多内存时扩展内存池。
      cuda_options.do_copy_in_default_stream = 1;                              // 行设置在默认流中进行复制操作为 1，这意味着 ONNX Runtime 将在 CUDA 的默认流中进行数据复制操作。
      cuda_options.has_user_compute_stream   = 0;                              // 这行设置用户计算流为 0，这意味着 ONNX Runtime 将使用其自己的计算流，而不是用户提供的计算流。
      cuda_options.default_memory_arena_cfg  = nullptr;                        // 这行设置默认内存区配置为 nullptr，这意味着 ONNX Runtime 将使用默认的内存区配置。

      session_options_extractor_.AppendExecutionProvider_CUDA( cuda_options );
      session_options_extractor_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED );
      session_options_matcher_.AppendExecutionProvider_CUDA( cuda_options );
      session_options_matcher_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED );
    }

    INFO( logger, "Loading extractor model from {0} and matcher model from {1}", config.extractor_path, config.matcher_path );
    session_uptr_extractor_ = std::make_unique<Ort::Session>( env_extractor_, config.extractor_path.c_str(), session_options_extractor_ );
    session_uptr_matcher_   = std::make_unique<Ort::Session>( env_matcher_, config.matcher_path.c_str(), session_options_matcher_ );

    // get input node names and shapes
    // extractor input node names and shapes
    INFO( logger, "extractor input node names and shapes" );
    extractNodesInfo( IO{ INPUT }, input_node_names_extractor_, input_node_shapes_extractor_, session_uptr_extractor_, allocator_ );
    INFO( logger, "extractor output node names and shapes" );
    extractNodesInfo( IO{ OUTPUT }, output_node_names_extractor_, output_node_shapes_extractor_, session_uptr_extractor_, allocator_ );

    // matcher input node names and shapes
    INFO( logger, "matcher input node names and shapes" );
    extractNodesInfo( IO{ INPUT }, input_node_names_matcher_, input_node_shapes_matcher_, session_uptr_matcher_, allocator_ );
    INFO( logger, "matcher output node names and shapes" );
    extractNodesInfo( IO{ OUTPUT }, output_node_names_matcher_, output_node_shapes_matcher_, session_uptr_matcher_, allocator_ );
  }
  catch ( const std::exception& e )
  {
    std::cerr << e.what() << '\n';
  }
}
