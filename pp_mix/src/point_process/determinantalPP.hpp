#ifndef DETERMINANTAL_PP_HPP
#define DETERMINANTAL_PP_HPP

//#include <stdexcept>
#include <omp.h>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>


#include "../rng.hpp"
#include "../utils.hpp"

using namespace Eigen;
using namespace stan::math;

class DeterminantalPP {
protected:
    MatrixXd ranges; //hyper-Rectangle R
    VectorXd diff_range;
    double vol_range; // |R|
    int dim; // dimension of space PP

    int N; // truncation of the eigendecomposition

    // parameters in the Gaussian spectral density
    double c; // >0 , it sets rho_max
    double s; // in (0,1), s.t. rho= s * rho_max

    // c_star is the integral of psi^* (in b&D algo), which is C_tilde(0)=sum(phis_tilde)
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

    // FOR PROPOSAL
     const MatrixXd * Lambda; // const pointer to the Lambda matrix

    double c_star_tmp; // c* of X (DPP) wrt the last passed Lambda

    // eigendecomposition params for last passed Lambda
    VectorXd phis_tmp;
    VectorXd phi_tildes_tmp;
    double Ds_tmp;

public:

    DeterminantalPP(const MatrixXd& ranges, int N, double c, double s);

    ~DeterminantalPP() {}

    // manages the decomposition in Ds, phi,... for the dpp (using or not lambda)
     // set the pointer to Lambda and performs the initial decomposition
    void set_decomposition(const MatrixXd * lambda);

    // modifies the passed Ds, phis, phi_tildes, c_star according to the dpp defined with lambda
    void compute_eigen_and_cstar(double * D_, VectorXd * Phis_, VectorXd * Phi_tildes_, double * C_star_, const MatrixXd * lambda);

    // manages decomposition wrt the passed lambda.
    // it takes the proposed Lambda and performs decomposition, storing it in "tmp" variables
    void decompose_proposal(const MatrixXd& lambda);

    void update_decomposition_from_proposal();

    void compute_Kappas(); // compute just once the grid for summation over Z^dim

    // computes (log default) density in x of cond process wrt the proposed matrix Lambda; USEFUL FOR MultiDpp.
    // For UniDpp it just calls dens_cond (because does not depend on Lambda), for MultiDpp it uses "tmp" variables.
    double dens_cond_in_proposal(const MatrixXd& x, bool log=true);

    // computes (log default) density in x of cond process wrt the current decomposition (expressed by the Father variables Ds, phis,...)
    double dens_cond(const MatrixXd& x, bool log=true);

    // computes (log default) density in x of dpp process wrt the current decomposition (expressed by the Father variables Ds, phis,...)
    double dens(const MatrixXd& x, bool log=true);


    // PRIVATE: final method that actually computes the log-density of dpp given parameters of decomposition
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


    // GETTERS

    MatrixXd get_ranges() const {return ranges;}

    double get_vol_range() const {return vol_range;}

    int get_dim() const {return dim;}

    double get_c() const {return c;}

    double get_s() const {return s;}

    double get_N() const {return N;}


    double get_cstar() const { return c_star; }

    const MatrixXd& get_kappas() const {return Kappas;}

    const VectorXd& get_phi_tildes() const {return phi_tildes;}

    const VectorXd& get_phis() const {return phis;}

    double get_Ds() const {return Ds;}

    const VectorXd& get_phi_tildes_tmp() const {return phi_tildes_tmp;}

    const VectorXd& get_phis_tmp() const {return phis_tmp;}

    double get_Ds_tmp() const {return Ds_tmp;}

    const MatrixXd& get_A() const {return A;}

    const VectorXd& get_b() const {return b;}

};

#endif
