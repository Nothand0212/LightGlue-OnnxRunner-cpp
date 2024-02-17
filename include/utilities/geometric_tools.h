#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>  // Include the necessary header file

class GeometricTools
{
public:
  bool static GeometricTools::calculate3DPointFromProjections( Eigen::Vector3f &projection_in_camera1, Eigen::Vector3f &projection_in_camera2, Eigen::Matrix<float, 3, 4> &projection_matrix_camera1, Eigen::Matrix<float, 3, 4> &projection_matrix_camera2, Eigen::Vector3f &point_3D )
  {
    Eigen::Matrix4f triangulation_matrix;
    triangulation_matrix.block<1, 4>( 0, 0 ) = projection_in_camera1( 0 ) * projection_matrix_camera1.block<1, 4>( 2, 0 ) - projection_matrix_camera1.block<1, 4>( 0, 0 );
    triangulation_matrix.block<1, 4>( 1, 0 ) = projection_in_camera1( 1 ) * projection_matrix_camera1.block<1, 4>( 2, 0 ) - projection_matrix_camera1.block<1, 4>( 1, 0 );
    triangulation_matrix.block<1, 4>( 2, 0 ) = projection_in_camera2( 0 ) * projection_matrix_camera2.block<1, 4>( 2, 0 ) - projection_matrix_camera2.block<1, 4>( 0, 0 );
    triangulation_matrix.block<1, 4>( 3, 0 ) = projection_in_camera2( 1 ) * projection_matrix_camera2.block<1, 4>( 2, 0 ) - projection_matrix_camera2.block<1, 4>( 1, 0 );

    Eigen::JacobiSVD<Eigen::Matrix4f> svd( triangulation_matrix, Eigen::ComputeFullV );

    Eigen::Vector4f homogeneous_point_3D = svd.matrixV().col( 3 );

    if ( homogeneous_point_3D( 3 ) == 0 )
    {
      return false;
    }

    // Euclidean coordinates
    point_3D = homogeneous_point_3D.head( 3 ) / homogeneous_point_3D( 3 );

    return true;
  }
};