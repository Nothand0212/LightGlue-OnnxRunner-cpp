#include <opencv2/opencv.hpp>

#include "base_onnx_runner.h"
#include "combined_onnx_runner.h"
#include "configuration.h"
#include "image_process.h"
#include "log/logger.h"
#include "visualizer.h"
std::vector<cv::Mat> readImage( std::vector<cv::String> image_file_vec, bool grayscale = false )
{
  /*
    Func:
        Read an image from path as RGB or grayscale

    */
  int mode = cv::IMREAD_COLOR;
  if ( grayscale )
  {
    mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
  }

  std::vector<cv::Mat> image_matlist;
  for ( const auto& file : image_file_vec )
  {
    logger->info( "Reading image: {0}", file );
    cv::Mat image = cv::imread( file, mode );
    if ( image.empty() )
    {
      throw std::runtime_error( "[ERROR] Could not read image at " + file );
    }
    if ( !grayscale )
    {
      cv::cvtColor( image, image, cv::COLOR_BGR2RGB );  // BGR -> RGB
    }
    image_matlist.emplace_back( image );
  }

  return image_matlist;
}

void visualizeMatches( const cv::Mat& src, const cv::Mat& dst, const std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& key_points, const std::vector<cv::Point2f>& key_points_src, const std::vector<cv::Point2f>& key_points_dst )
{
  // Convert the points to cv::KeyPoint objects
  std::vector<cv::KeyPoint> key_points_1, key_points_2;
  for ( const auto& point : key_points.first )
  {
    key_points_1.push_back( cv::KeyPoint( point, 1.0f ) );
  }
  for ( const auto& point : key_points.second )
  {
    key_points_2.push_back( cv::KeyPoint( point, 1.0f ) );
  }

  // Create cv::DMatch objects for each pair of points
  std::vector<cv::DMatch> matches;
  for ( size_t i = 0; i < key_points_1.size(); ++i )
  {
    matches.push_back( cv::DMatch( i, i, 0 ) );
  }

  // Draw the matches
  cv::Mat img_matches;
  cv::drawMatches( src, key_points_1, dst, key_points_2, matches, img_matches );

  for ( const auto& point : key_points_src )
  {
    cv::circle( img_matches, point, 5, cv::Scalar( 0, 0, 255 ), -1 );
  }

  for ( const auto& point : key_points_dst )
  {
    cv::circle( img_matches, cv::Point2f( point.x + src.cols, point.y ), 5, cv::Scalar( 0, 0, 255 ), -1 );
  }

  // Display the matches
  cv::imshow( "Matches", img_matches );
  cv::waitKey( 0 );
}

int main( int argc, char const* argv[] )
{
  InitLogger( "/home/lin/CLionProjects/light_glue_onnx/log/tmp.log" );
  Config cfg{};
  cfg.readConfig( "/home/lin/CLionProjects/light_glue_onnx/config/param.json" );

  std::vector<cv::String> image_file_src_vec;
  std::vector<cv::String> image_file_dst_vec;

  // Read image file path
  cv::glob( cfg.image_src_path, image_file_src_vec );
  cv::glob( cfg.image_dst_path, image_file_dst_vec );

  // Read image
  if ( image_file_src_vec.size() != image_file_dst_vec.size() )
  {
    logger->error( "image src number: {0}", image_file_src_vec.size() );
    logger->error( "image dst number: {0}", image_file_dst_vec.size() );
    throw std::runtime_error( "[ERROR] The number of images in the left and right folders is not equal" );
    return EXIT_FAILURE;
  }

  std::vector<cv::Mat> image_src_mat_vec = readImage( image_file_src_vec, cfg.gray_flag );
  std::vector<cv::Mat> image_dst_mat_vec = readImage( image_file_dst_vec, cfg.gray_flag );

  // end2end
  CombinedOnnxRunner* feature_matcher;
  feature_matcher = new CombinedOnnxRunner{ 0 };
  feature_matcher->initOrtEnv( cfg );
  feature_matcher->setMatchThreshold( cfg.threshold );

  // inference
  auto iter_src = image_src_mat_vec.begin();
  auto iter_dst = image_dst_mat_vec.begin();
  for ( ; iter_src != image_src_mat_vec.end(); ++iter_src, ++iter_dst )
  {
    auto key_points_result = feature_matcher->inferenceImagePair( cfg, *iter_src, *iter_dst );

    auto key_points_src = feature_matcher->getKeyPointsSrc();
    auto key_points_dst = feature_matcher->getKeyPointsDst();

    for ( const auto& point : key_points_src )
    {
      cv::circle( *iter_src, point, 5, cv::Scalar( 0, 0, 255 ), -1 );
    }

    for ( const auto& point : key_points_dst )
    {
      cv::circle( *iter_dst, point, 5, cv::Scalar( 0, 0, 255 ), -1 );
    }

    visualizeMatches( *iter_src, *iter_dst, key_points_result, key_points_src, key_points_dst );

    // break;
  }
  return 0;
}
