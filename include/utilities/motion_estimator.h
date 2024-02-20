#pragma once

#include "frame/frame.h"
#include "frame/key_frame.h"

class MotionEstimator
{
private:
  /* data */
public:
  MotionEstimator( /* args */ );
  ~MotionEstimator();
  // PnP-based motion estimation. Accepts one frame's map-points and other one frame's 2D points and returns the transformation between them
  bool static calculateMotionFrom3D2D();
  // ICP-based motion estimation. Accepts two frames map-points and returns the transformation between them
  bool static calculateMotionFrom3D3D();
};

MotionEstimator::MotionEstimator( /* args */ )
{
}

MotionEstimator::~MotionEstimator()
{
}
