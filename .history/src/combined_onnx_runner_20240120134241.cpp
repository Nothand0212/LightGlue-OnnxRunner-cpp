/**
 ******************************************************************************
 * @file           : include/combined_onnx_runner.cpp
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-18
 ******************************************************************************
 */

#include "combined_onnx_runner.h"
#include "log/logger.h"

CombinedOnnxRunner::CombinedOnnxRunner( unsigned int threads_num ) : threads_num_( threads_num )
{
  logger->info( "CombinedOnnxRunner created" );
}

CombinedOnnxRunner::~CombinedOnnxRunner()
{
  logger->info( "CombinedOnnxRunner destroyed" );

  for ( char* input_node_name : input_node_names_ )
  {
    delete input_node_name;
  }
  input_node_names_.clear();
}


int CombinedOnnxRunner::initOrtEnv( const Config& config )
{
  logger->info( "initOrtEnv start" );

  try
  {
    env_ = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "CombinedOnnxRunner" );

    // create session options
    session_options_ = Ort::SessionOptions();
    if ( threads_num_ == 0 )
    {
      threads_num_ = std::thread::hardware_concurrency();
    }
    session_options_.SetIntraOpNumThreads( threads_num_ );
    session_options_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    logger->info( "Using {0} threads, with graph optimization level {1}", threads_num_, GraphOptimizationLevel::ORT_ENABLE_ALL );


    if ( config.device == "cuda" )
    {
      logger->info( "Using CUDA provider with default options" );

      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id                 = 0;                              // 这行设置 CUDA 设备 ID 为 0，这意味着 ONNX Runtime 将在第一个 CUDA 设备（通常是第一个 GPU）上运行模型。
      cuda_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchDefault;  // 这行设置 cuDNN 卷积算法搜索策略为默认值。cuDNN 是 NVIDIA 的深度神经网络库，它包含了许多用于卷积的优化算法。
      cuda_options.gpu_mem_limit             = 0;                              // 这行设置 GPU 内存限制为 0，这意味着 ONNX Runtime 可以使用所有可用的 GPU 内存。
      cuda_options.arena_extend_strategy     = 1;                              // 这行设置内存分配策略为 1，这通常意味着 ONNX Runtime 将在需要更多内存时扩展内存池。
      cuda_options.do_copy_in_default_stream = 1;                              // 行设置在默认流中进行复制操作为 1，这意味着 ONNX Runtime 将在 CUDA 的默认流中进行数据复制操作。
      cuda_options.has_user_compute_stream   = 0;                              // 这行设置用户计算流为 0，这意味着 ONNX Runtime 将使用其自己的计算流，而不是用户提供的计算流。
      cuda_options.default_memory_arena_cfg  = nullptr;                        // 这行设置默认内存区配置为 nullptr，这意味着 ONNX Runtime 将使用默认的内存区配置。

      session_options_.AppendExecutionProvider_CUDA( cuda_options );
      session_options_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED );
    }

    // const char* model_path = config.combiner_path.c_str();
    logger->info( "Loading model from {0}", config.combiner_path );
    session_uptr_ = std::make_unique<Ort::Session>( env_, config.combiner_path.c_str(), session_options_ );

    // get input node names

    const size_t num_input_nodes = session_uptr_->GetInputCount();
    input_node_names_.reserve( num_input_nodes );
    for ( size_t i = 0; i < num_input_nodes; i++ )
    {
      char* input_node_name_temp = new char[ std::strlen( session_uptr_->GetInputNameAllocated( i, allocator_ ).get() ) + 1 ];
      std::strcpy( input_node_name_temp, session_uptr_->GetInputNameAllocated( i, allocator_ ).get() );
      logger->info( "input node name: {0}", input_node_name_temp );
      // input_node_names_[ i ] = input_node_name_temp;
      input_node_names_.push_back( input_node_name_temp );

      //   input_node_names_.push_back( std::string( session_uptr_->GetInputNameAllocated( i, allocator_ ).get() ).c_str() );
      input_node_shapes_.emplace_back( session_uptr_->GetInputTypeInfo( i ).GetTensorTypeAndShapeInfo().GetShape() );
    }

    const size_t num_output_nodes = session_uptr_->GetOutputCount();
    output_node_names_.reserve( num_output_nodes );
    for ( size_t i = 0; i < num_output_nodes; i++ )
    {
      char* output_node_names_temp = new char[ std::strlen( session_uptr_->GetOutputNameAllocated( i, allocator_ ).get() ) + 1 ];
      std::strcpy( output_node_names_temp, session_uptr_->GetOutputNameAllocated( i, allocator_ ).get() );
      logger->info( "output node name: {0}", output_node_names_temp );
      // output_node_names_[ i ] = output_node_names_temp;
      output_node_names_.push_back( output_node_names_temp );


      //   output_node_names_.push_back( std::string( session_uptr_->GetOutputNameAllocated( i, allocator_ ).get() ).c_str() );
      output_node_shapes_.emplace_back( session_uptr_->GetOutputTypeInfo( i ).GetTensorTypeAndShapeInfo().GetShape() );
    }


    // delete model_path;
    logger->info( "ONNX Runtime environment initialized successfully!" );
    logger->info( "input node names size: {0}", input_node_names_.size() );
  }
  catch ( const std::exception& e )
  {
    logger->error( "ONNX Runtime environment initialized failed! Error message: {0}", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

cv::Mat CombinedOnnxRunner::preProcess( const Config& config, const cv::Mat& image_src, float& scale )
{
  float scale_temp = scale;

  cv::Mat image_temp = image_src.clone();
  logger->info( "image_temp size: [{0}, {1}]", image_temp.cols, image_temp.rows );

  std::string fn{ "max" };
  std::string interp{ "area" };
  cv::Mat     image_result = normalizeImage( resizeImage( image_temp, config.image_size, scale, fn, interp ) );

  logger->info( "image_result size: [{0}, {1}], scale from {2} to {3}", image_result.cols, image_result.rows, scale_temp, scale );

  return image_result;
}

int CombinedOnnxRunner::inference( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst )
{
  logger->info( "**** inference start ****" );

  try
  {
    logger->info( "image_src shape: [{0}, {1}], channel: {2}", image_src.cols, image_src.rows, image_src.channels() );
    logger->info( "image_dst shape: [{0}, {1}], channel: {2}", image_dst.cols, image_dst.rows, image_dst.channels() );

    logger->info( "input node names size: {0}", input_node_names_.size() );
    for ( const auto& name : input_node_names_ )
    {
      logger->info( "****input node name: {0}", name );
    }

    logger->info( "output node names size: {0}", output_node_names_.size() );
    for ( const auto& name : output_node_names_ )
    {
      logger->info( "****output node name: {0}", name );
    }

    // build source input node shape and destination input node shape
    // only support super point
    logger->info( "create input tensors" );
    input_node_shapes_[ 0 ]   = { 1, 1, image_src.size().height, image_src.size().width };
    input_node_shapes_[ 1 ]   = { 1, 1, image_dst.size().height, image_dst.size().width };
    int input_tensor_size_src = input_node_shapes_[ 0 ][ 0 ] * input_node_shapes_[ 0 ][ 1 ] * input_node_shapes_[ 0 ][ 2 ] * input_node_shapes_[ 0 ][ 3 ];
    int input_tensor_size_dst = input_node_shapes_[ 1 ][ 0 ] * input_node_shapes_[ 1 ][ 1 ] * input_node_shapes_[ 1 ][ 2 ] * input_node_shapes_[ 1 ][ 3 ];

    std::vector<float> input_tensor_values_src( input_tensor_size_src );
    std::vector<float> input_tensor_values_dst( input_tensor_size_dst );

    input_tensor_values_src.assign( image_src.begin<float>(), image_src.end<float>() );
    input_tensor_values_dst.assign( image_dst.begin<float>(), image_dst.end<float>() );

    // create input tensor object from data values
    logger->info( "create memory info handler" );
    auto memory_info_handler = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU );

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, input_tensor_values_src.data(), input_tensor_size_src, input_node_shapes_[ 0 ].data(), input_node_shapes_[ 0 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, input_tensor_values_dst.data(), input_tensor_size_dst, input_node_shapes_[ 1 ].data(), input_node_shapes_[ 1 ].size() ) );

    logger->info( "run inference" );
    auto timer_tic          = std::chrono::high_resolution_clock::now();
    auto output_tensor_temp = session_uptr_->Run( Ort::RunOptions{ nullptr }, input_node_names_.data(), input_tensors.data(), input_tensors.size(), output_node_names_.data(), output_node_names_.size() );
    auto timer_toc          = std::chrono::high_resolution_clock::now();
    auto diff               = std::chrono::duration_cast<std::chrono::milliseconds>( timer_toc - timer_tic ).count();
    timer_ += diff;


    int count = 0;
    for ( const auto& tensor : output_tensor_temp )
    {
      if ( !tensor.IsTensor() )
      {
        logger->error( "Output {0} is not a tensor", count );
      }

      if ( !tensor.HasValue() )
      {
        logger->error( "Output {0} is empty", count );
      }

      count++;
    }

    output_tensors_ = std::move( output_tensor_temp );
    logger->info( "inference finished!" );
    logger->info( "inference time: {0} / {1} ms", diff, timer_ );
  }
  catch ( const std::exception& e )
  {
    logger->error( "**** inference failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


int CombinedOnnxRunner::postProcess( const Config& config )
{
  try
  {
    logger->info( "**** postProcess start ****" );
    logger->info( "output tensors size: {0}", output_tensors_.size() );

    std::vector<int64_t> key_points_0_shape = output_tensors_[ 0 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [1, num_key_points, 2]
    int64_t*             key_points_0_ptr   = (int64_t*)output_tensors_[ 0 ].GetTensorMutableData<void>();
    logger->info( "key points 0 shape: [{0}, {1}, {2}], size: {3}", key_points_0_shape[ 0 ], key_points_0_shape[ 1 ], key_points_0_shape[ 2 ], key_points_0_shape.size() );

    std::vector<int64_t> key_points_1_shape = output_tensors_[ 1 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [1, num_key_points, 2]
    int64_t*             key_points_1_ptr   = (int64_t*)output_tensors_[ 1 ].GetTensorMutableData<void>();
    logger->info( "key points 1 shape: [{0}, {1}, {2}], size: {3}", key_points_1_shape[ 0 ], key_points_1_shape[ 1 ], key_points_1_shape[ 2 ], key_points_1_shape.size() );


    std::vector<int64_t> matches_0_shape = output_tensors_[ 2 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [num_matches, 2]
    int64_t*             matches_0_ptr   = (int64_t*)output_tensors_[ 2 ].GetTensorMutableData<void>();
    logger->info( "matches 0 shape: [{0}, {1}], size: {2}", matches_0_shape[ 0 ], matches_0_shape[ 1 ], matches_0_shape.size() );

    std::vector<int64_t> match_scores_0_shape = output_tensors_[ 3 ].GetTensorTypeAndShapeInfo().GetShape();  // shape: [num_matches]
    float*               match_scores_0_ptr   = (float*)output_tensors_[ 3 ].GetTensorMutableData<void>();
    logger->info( "matches score shape: [{0}, {1}], size: {2}", match_scores_0_shape[ 0 ], match_scores_0_shape[ 1 ], match_scores_0_shape.size() );

    // process key points
    std::vector<cv::Point2f> key_points_0_tmp, key_points_1_tmp;
    for ( int i = 0; i < key_points_0_shape[ 1 ] * 2; i += 2 )
    {
      key_points_0_tmp.emplace_back( cv::Point2f( ( key_points_0_ptr[ i ] + 0.5f ) / scales_[ 0 ] - 0.5f, ( key_points_0_ptr[ i + 1 ] + 0.5f ) / scales_[ 0 ] - 0.5f ) );
    }
    for ( int i = 0; i < key_points_1_shape[ 1 ] * 2; i += 2 )
    {
      key_points_1_tmp.emplace_back( cv::Point2f( ( key_points_1_ptr[ i ] + 0.5f ) / scales_[ 1 ] - 0.5f, ( key_points_1_ptr[ i + 1 ] + 0.5f ) / scales_[ 1 ] - 0.5f ) );
    }


    // create matches indices
    std::set<std::pair<int, int>> matches;
    for ( int i = 0; i < matches_0_shape[ 0 ]; i += 2 )
    {
      if ( matches_0_ptr[ i ] > -1 && match_scores_0_ptr[ i ] > match_threshold_ )
      {
        logger->info( "matche pair index: [{0}, {1}]", matches_0_ptr[ i ], matches_0_ptr[ i + 1 ] );
        matches.insert( std::make_pair( matches_0_ptr[ i ], matches_0_ptr[ i + 1 ] ) );
      }
    }
    logger->info( "matches size: {0}", matches.size() );


    std::vector<cv::Point2f> key_points_0, key_points_1;


    for ( const auto& match : matches )
    {
      key_points_0.emplace_back( key_points_0_tmp[ match.first ] );
      key_points_1.emplace_back( key_points_1_tmp[ match.second ] );
    }

    key_points_result_.first  = key_points_0;
    key_points_result_.second = key_points_1;

    logger->info( "**** postProcess success ****" );
  }
  catch ( const std::exception& e )
  {
    logger->error( "**** postProcess failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
CombinedOnnxRunner::inferenceImagePair( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst )
{
  logger->info( "**** inferenceImage start ****" );

  if ( image_src.empty() || image_dst.empty() )
  {
    logger->error( "inferenceImage failed, image is empty." );
    throw std::runtime_error( "image is empty" );
  }

  cv::Mat image_src_copy = image_src.clone();
  cv::Mat image_dst_copy = image_dst.clone();

  logger->info( "preprocessing image_src" );
  cv::Mat image_src_temp = preProcess( config, image_src_copy, scales_[ 0 ] );
  logger->info( "preprocessing image_dst" );
  cv::Mat image_dst_temp = preProcess( config, image_dst_copy, scales_[ 1 ] );


  int inference_result = inference( config, image_src_temp, image_dst_temp );
  if ( inference_result != EXIT_SUCCESS )
  {
    logger->error( "**** inferenceImage failed ****" );
    throw std::runtime_error( "inference failed" );
  }

  int post_process_result = postProcess( config );
  if ( post_process_result != EXIT_SUCCESS )
  {
    logger->error( "**** inferenceImage failed ****" );
    return std::make_pair( std::vector<cv::Point2f>(), std::vector<cv::Point2f>() );
  }

  output_tensors_.clear();
  logger->info( "**** inferenceImage success ****" );
  return getKeyPointsResult();
}

float CombinedOnnxRunner::getMatchThreshold() const
{
  return match_threshold_;
}

void CombinedOnnxRunner::setMatchThreshold( float match_threshold )
{
  match_threshold_ = match_threshold;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> CombinedOnnxRunner::getKeyPointsResult() const
{
  return key_points_result_;
}