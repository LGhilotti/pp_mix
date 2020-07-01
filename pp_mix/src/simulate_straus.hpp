#ifndef SIMULATE_STRAUSS_HPP
#define SIMULATE_STRAUSS_HPP

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <stdlib.h>

#include <Eigen/Dense>

#include "point_process/strauss_pp.hpp"
#include "pointprocess/point2pattern.h"
#include "pointprocess/pointprocess.h"
#include "pointprocess/sampler.h"
#include "pointprocess/strauss.h"
#include "rng.hpp"

using namespace Eigen;

Eigen::MatrixXd simulate_strauss_moller(const Eigen::MatrixXd &ranges,
                                        double beta, double gamma, double R);

Eigen::MatrixXd simulate_strauss_our(StraussPP *pp);

std::tuple<MatrixXd, std::vector<VectorXd>, std::vector<VectorXd>, VectorXd,
           VectorXi, VectorXi>
run_backwards(const MatrixXd &init, int nsteps, StraussPP *pp);

#endif