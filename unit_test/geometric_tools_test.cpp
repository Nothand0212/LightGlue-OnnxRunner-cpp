#include <gtest/gtest.h>

#include "geometric_tools.h"

TEST( GeometricToolsTest, Calculate3DPointFromProjections )
{
  Eigen::Vector3f            projection_in_camera1( 1, 2, 3 );
  Eigen::Vector3f            projection_in_camera2( 4, 5, 6 );
  Eigen::Matrix<float, 3, 4> projection_matrix_camera1 = Eigen::Matrix<float, 3, 4>::Random();
  Eigen::Matrix<float, 3, 4> projection_matrix_camera2 = Eigen::Matrix<float, 3, 4>::Random();
  Eigen::Vector3f            point_3D;

  bool result = GeometricTools::calculate3DPointFromProjections( projection_in_camera1, projection_in_camera2, projection_matrix_camera1, projection_matrix_camera2, point_3D );

  // Replace with your expected results
  bool            expected_result = true;
  Eigen::Vector3f expected_point_3D( 1, 1, 1 );

  EXPECT_EQ( result, expected_result );
  EXPECT_EQ( point_3D, expected_point_3D );
}