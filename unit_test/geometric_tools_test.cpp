#include <gtest/gtest.h>

#include "geometric_tools.h"
TEST( GeometricToolsTest, Calculate3DPointFromProjections )
{
  // Define the projection matrices for two cameras
  Eigen::Matrix<float, 3, 4> projection_matrix_camera1;
  projection_matrix_camera1 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0;
  Eigen::Matrix<float, 3, 4> projection_matrix_camera2;
  projection_matrix_camera2 << 1, 0, 0, -1,
      0, 1, 0, 0,
      0, 0, 1, 0;

  // Define a known 3D point
  Eigen::Vector4f point_3D_known( 1, 2, 3, 1 );

  // Project the known 3D point into the two cameras
  Eigen::Vector3f projection_in_camera1 = projection_matrix_camera1 * point_3D_known;
  Eigen::Vector3f projection_in_camera2 = projection_matrix_camera2 * point_3D_known;

  // Normalize the projections to get the 2D points
  projection_in_camera1 /= projection_in_camera1( 2 );
  projection_in_camera2 /= projection_in_camera2( 2 );

  // Use the 2D points and the camera matrices to triangulate the 3D point
  Eigen::Vector3f point_3D;
  bool            result = GeometricTools::calculate3DPointFromProjections( projection_in_camera1, projection_in_camera2, projection_matrix_camera1, projection_matrix_camera2, point_3D );

  // Check that the triangulated point is close to the known point
  EXPECT_TRUE( result );
  EXPECT_NEAR( point_3D( 0 ), point_3D_known( 0 ), 1e-5 );
  EXPECT_NEAR( point_3D( 1 ), point_3D_known( 1 ), 1e-5 );
  EXPECT_NEAR( point_3D( 2 ), point_3D_known( 2 ), 1e-5 );
}