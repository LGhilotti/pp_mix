#ifndef BASE_PP_HPP
#define BASE_PP_HPP

#include <stdexcept>
#include <Eigen/Dense>
#include <stan/math/prim/mat.hpp>
#include <google/protobuf/message.h>

#include "../rng.hpp"
#include "../utils.hpp"
#include "../../protos/cpp/state.pb.h"
#include "point.hpp"

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

    virtual void initialize()  = 0;

    virtual double dens(const MatrixXd& x, bool log=true) = 0;

    virtual double papangelou(
        MatrixXd xi, const MatrixXd &x, bool log = true) = 0;

    virtual double papangelou(const Point &xi, const std::list<Point> &x,
                              bool log = true) = 0;

    MatrixXd sample_uniform(int npoints);

    // Sample from b_bar'(csi)=phi_star(csi)/c_star (used in birth process)
    virtual VectorXd phi_star_rng() = 0;

    // Gives (log) of phi_star(xi)
    virtual double phi_star_dens(VectorXd xi, bool log = true) = 0;

    void sample_given_active(
        const MatrixXd &active, MatrixXd *non_active, double psi_u);

    // PERFECT SIMULATION
    virtual void update_hypers(const MatrixXd &active, const MatrixXd &non_active) = 0;

    virtual void get_state_as_proto(google::protobuf::Message *out) = 0;

    virtual double rejection_sampling_M(int npoints) = 0;

    /*
    * this method is called by ConditionalMCMC::sample_means()
    * it should return a value for the standard deviation of the proposal
    * density in such a way that it gives sufficient high probability
    * to regions where the repulsion is lower.
    */
    virtual double estimate_mean_proposal_sigma() = 0;

    MatrixXd get_ranges() const {return ranges;}

    double get_vol_range() const {return vol_range;}

    int get_dim() const {return ranges.cols();}

    double get_cstar() const { return c_star; }

    MatrixXd sample_n_points(int npoints);
};

#endif
