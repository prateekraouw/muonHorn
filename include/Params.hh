// include/Params.hh
#pragma once
struct HornParams {
  // Geometry shared by both horns
  double a_mm, Rout_mm, r_neck_mm;
  double zMin_mm, zMax_mm;
  double r_max_mm;
  // Current magnitude (A); Horn1:+I, Horn2:-I
  double I_A;
  // Spacing between horns (mm)
  double spacing_mm;
  // Run control
  int n_events = 10000;
};

