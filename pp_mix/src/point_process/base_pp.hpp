#ifndef BASE_PP_HPP
#define BASE_PP_HPP

#include <Eigen/Dense>
#include <stan/math/prim/mat.hpp>
#include <google/protobuf/message.h>

#include "../rng.hpp"
#include "../utils.hpp"
#include "../../protos/cpp/state.pb.h"

using namespace Eigen;
using namespace stan::math;


class BasePP {

 protected:
    MatrixXd ranges;
    VectorXd diff_range;
    double vol_range;
    int dim;
    double c_star;

    // FOR DEBUGGING
    double birth_prob, birth_arate;

 public:
    BasePP() {}

    BasePP(const MatrixXd& ranges);

    void set_ranges(const MatrixXd& ranges);

    virtual ~BasePP() {}

    virtual double dens(const MatrixXd& x, bool log=true) = 0;

    virtual double papangelou(
        MatrixXd xi, const MatrixXd &x, bool log = true) = 0;

    MatrixXd sample_uniform(int npoints);

    virtual VectorXd phi_star_rng() = 0;

    virtual double phi_star_dens(VectorXd xi, bool log = true) = 0;

    void sample_given_active(
        const MatrixXd &active, MatrixXd *non_active, double psi_u);

    virtual void update_hypers(const MatrixXd &active, const MatrixXd &non_active) = 0;

    virtual void get_state_as_proto(google::protobuf::Message *out) = 0;

    MatrixXd get_ranges() const {return ranges;}
};

#endif