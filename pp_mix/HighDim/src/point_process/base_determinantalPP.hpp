#ifndef BASE_DETERMINANTAL_PP_HPP
#define BASE_DETERMINANTAL_PP_HPP

//#include <stdexcept>
#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <google/protobuf/message.h>

#include "../rng.hpp"
#include "../utils.hpp"
//#include "../../protos/cpp/state.pb.h"

using namespace Eigen;
using namespace stan::math;

class BaseDeterminantalPP {
protected:
    MatrixXd ranges; //hyper-Rectangle R
    VectorXd diff_range;
    double vol_range; // |R|
    int dim; // dimension of space PP

    int N; // truncation of the eigendecomposition

    // parameters in the Gaussian spectral density
    double c; // >0 , it sets rho_max
    double s; // in (0,1), s.t. rho= s * rho_max

    double c_star; // c* of X (DPP): changes when Lambda changes!

    // eigendecomposition params: change when Lambda changes!
    VectorXd phis;
    VectorXd phi_tildes;
    double Ds;

    MatrixXd Kappas; // grid for approximating summation over Z^dim

    // Affine transformation to the unit square
    MatrixXd A;
    VectorXd b;

    // FOR DEBUGGING
    double birth_prob, birth_arate;

public:

    BaseDeterminantalPP(const MatrixXd& ranges, int N, double c, double s);

    virtual ~BaseDeterminantalPP() {}

    // manages the decomposition in Ds, phi,... for the dpp (using or not lambda)
    virtual void set_decomposition(const MatrixXd * lambda) = 0;

    // manages decomposition wrt the passed lambda; ONLY USED FOR MultiDpp
    // I define it doing nothing, so UniDpp inherits it; override in MultiDpp
    virtual void decompose_proposal(const MatrixXd& lambda) {} ;

    virtual void compute_Kappas() = 0; // compute just once the grid for summation over Z^dim

    // computes (log default) density in x of cond process wrt the proposed matrix Lambda; USEFUL FOR MultiDpp.
    // For UniDpp it just calls dens_cond (because does not depend on Lambda), for MultiDpp it uses "tmp" variables.
    virtual double dens_cond_in_proposal(const MatrixXd& x, bool log=true) = 0;

    // computes (log default) density in x of cond process wrt the current decomposition (expressed by the Father variables Ds, phis,...)
    double dens_cond(const MatrixXd& x, bool log=true);

    // computes (log default) density in x of dpp process wrt the current decomposition (expressed by the Father variables Ds, phis,...)
    double dens(const MatrixXd& x, bool log=true);


    // final method that actually computes the log-density of dpp given parameters of decomposition
    double ln_dens_process(const MatrixXd& x, double Ds_p, const VectorXd& phis_p, const VectorXd& phi_tildes_p, double c_star_p);



    double log_det_Ctilde(const MatrixXd& x, const VectorXd& phi_tildes_p);

    double papangelou(const VectorXd& xi, const MatrixXd &x, bool log = true);

    // sample npoints uniformly in the region defined by range
    MatrixXd sample_uniform(int npoints);

    // Sample from b_bar'(csi)=phi_star(csi)/c_star (used in birth process)
    VectorXd phi_star_rng();

    // Gives (log) of phi_star(xi)
    double phi_star_dens(VectorXd xi, bool log = true);

    // Sample B&D from posterior of mu_na
    void sample_nonalloc_fullcond(
         MatrixXd *non_active, const MatrixXd &active, double psi_u);

    //void get_state_as_proto(google::protobuf::Message *out) {}

    MatrixXd get_ranges() const {return ranges;}

    double get_vol_range() const {return vol_range;}

    int get_dim() const {return ranges.cols();}

    double get_cstar() const { return c_star; }

    MatrixXd get_kappas() const {return Kappas;}

};

#endif
