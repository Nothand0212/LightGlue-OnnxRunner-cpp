/**
 ******************************************************************************
 * @file           : include/visualizer.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-20
 ******************************************************************************
 */

#pragma once

#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

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
    cv::circle( img_matches, point, 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  for ( const auto& point : key_points_dst )
  {
    cv::circle( img_matches, cv::Point2f( point.x + src.cols, point.y ), 2, cv::Scalar( 0, 0, 255 ), -1 );
  }

  // Display the matches
  cv::imshow( "Matches", img_matches );
  cv::waitKey( 0 );
}