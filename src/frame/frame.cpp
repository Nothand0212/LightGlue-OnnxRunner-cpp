#include "frame/frame.h"

std::size_t Frame::m_id_counter = 0;

Frame::Frame( const cv::Mat& image_left, const cv::Mat& image_right, const double& time_stamp )
    : m_image_left( image_left ), m_image_right( image_right ), m_time_stamp( time_stamp )
{
  m_id = m_id_counter++;
}
