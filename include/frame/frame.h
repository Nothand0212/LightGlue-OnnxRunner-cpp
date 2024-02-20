#pragma once


#include <eigen3/Eigen/Core>
#include <mutex>
#include <opencv2/core.hpp>
#include <sophus/so3.hpp>
#include <vector>

#include "log/logger.h"
#include "map_point/map_point.h"

class Frame
{
private:
  static std::size_t m_id_counter;

  cv::Mat                  m_image_left, m_image_right;              // Left and right images
  std::vector<cv::Point2f> m_key_points_left, m_key_points_right;    // Keypoints in left and right images
  cv::Mat                  m_descriptors_left, m_descriptors_right;  // Descriptors for left and right images
  std::vector<MapPoint>    m_map_points;                             // Map points in the frame, constructed from key points triangulation

  // Pose related
  // T = [R | t] means T is the transformation
  Sophus::SE3d m_T_c_w;   // T_c_w is the transformation from world to camera
  Sophus::SE3d m_T_c_kf;  // T_c_kf is the transformation from keyframe to camera

  std::mutex m_pose_w_mutex;
  std::mutex m_pose_kf_mutex;

  std::size_t m_id;          // Frame id
  double      m_time_stamp;  // Time stamp of the frame


public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Frame() = default;
  Frame( const cv::Mat& image_left, const cv::Mat& image_rightm, const double& time_stamp );
  ~Frame() = default;

  // Setters
  void setKeyPoints( const std::vector<cv::Point2f>& key_points_left, const std::vector<cv::Point2f>& key_points_right );
  void setKeyPointsLeft( const std::vector<cv::Point2f>& key_points_left );
  void setKeyPointsRight( const std::vector<cv::Point2f>& key_points_right );
  void setDescriptors( const cv::Mat& descriptors_left, const cv::Mat& descriptors_right );
  void setDescriptorsLeft( const cv::Mat& descriptors_left );
  void setDescriptorsRight( const cv::Mat& descriptors_right );

  void setPose( const Sophus::SE3d& T_c_w );
  void setRelativePose( const Sophus::SE3d& T_c_kf );
};
