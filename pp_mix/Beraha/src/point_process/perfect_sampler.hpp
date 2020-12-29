#ifndef PERFECT_SAMPLER_HPP
#define PERFECT_SAMPLER_HPP

#include <Eigen/Dense>
#include <map>
#include <deque>

#include "base_pp.hpp"
#include "point.hpp"

using namespace Eigen;

class PerfectSampler {
 protected:
  BasePP* pp;
  std::map<int, Point> id2point;
  std::deque<int> in_lower;
  std::deque<int> in_upper;
  std::deque<std::tuple<int, bool, double>> transitions;
  std::deque<Point> state;
  int max_id = 0;
  int numupper = 0;
  int numlower = 0;
  int double_t = 2;

 public:
  PerfectSampler() {}

  PerfectSampler(BasePP* pp) : pp(pp) {}

  ~PerfectSampler() {}

  void initialize();

  void estimate_doubling_time();

  void one_backward(std::deque<Point>* points);

  void one_forward(std::tuple<int, bool, double> trans);

  MatrixXd simulate();
};

#endif