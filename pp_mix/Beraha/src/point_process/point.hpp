#ifndef POINT_HPP
#define POINT_HPP

#include <Eigen/Dense>
using namespace Eigen;

struct Point {
  VectorXd coords;
  double r_mark;
  int number = -100;  // each point has associated an unique id
};

#endif 