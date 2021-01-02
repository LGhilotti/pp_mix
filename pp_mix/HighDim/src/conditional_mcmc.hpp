#ifndef CONDITIONAL_MCMC
#define CONDITIONAL_MCMC

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <functional>

#include <Eigen/Dense>
#include <stan/math/prim.hpp>
#include <google/protobuf/message.h>

#include "gig.hpp"
#include "rng.hpp"
#include "point_process/base_determinantalPP.hpp"
#include "precs/base_prec.hpp"
#include "precs/precmat.hpp"
#include "utils.hpp"
//#include "../protos/cpp/state.pb.h"
//#include "../protos/cpp/params.pb.h"


using namespace Eigen;


template<class Prec, typename prec_t, typename fact_t>
class ConditionalMCMC {
 protected:
     int dim_fact;
     int dim_data;
     int ndata;
     MatrixXd data;
     // fixed hyperparameters
     double _a_phi; // Dirichlet parameter
     double _alpha_jump, _beta_jump; // Gamma jump parameters
     double _a_gamma, _b_gamma; // Sigma_bar parameters

     // for each allocated cluster, it contains the vector of indexes of observations:
     // both useful for data and etas.
     std::vector<std::vector<int>> obs_by_clus;
     std::vector<std::vector<VectorXd>> etas_by_clus;

     /* STATE */
     // Rep-pp
     double u;
     VectorXi clus_alloc;
     VectorXd a_jumps, na_jumps;
     MatrixXd a_means, na_means;
     std::vector<prec_t> a_deltas, na_deltas;
     // etas: n x d matrix
     MatrixXd etas;
     //Sigma_bar
     VectorXd sigma_bar;
     //Lambda-block
     double tau;
     MatrixXd Phi;
     MatrixXd Psi;
     MatrixXd Lambda;

     // DISTRIBUTIONS
     BaseDeterminantalPP *pp_mix;
     Prec *g;

     // FOR DEBUGGING
     bool verbose = false;
     int acc_mean = 0;
     int tot_mean = 0;

     Params params;

     double min_proposal_sigma, max_proposal_sigma, prop_lambda_sigma;

 public:
     ConditionalMCMC() {}
     ~ConditionalMCMC()
     {
         delete pp_mix;
         delete g;
    }

    ConditionalMCMC(
        BaseDeterminantalPP * pp_mix, Prec * g,
        const Params& params);

    void set_pp_mix(BaseDeterminantalPP* pp_mix) {this->pp_mix = pp_mix;}
    void set_prec(Prec* g) {this->g = g;}
    void set_params(const Params & p);

    // initializes some of the members (data, dim, ndata,..) and state of sampler
    // The constructor only initialize some other members (pointers) and params field
    void initialize(const MatrixXd &dat);

    // initializes the etas, projecting the data onto Col(Lambda):
    // it is for both uni/multi factor cases, but implemented differently because of the least square systems.
    virtual void initialize_etas(const MatrixXd &dat) = 0;

    virtual void initialize_allocated_means() = 0;

    std::vector<VectorXd> proj_inside();
    bool is_inside(const VectorXd & eta);

    // it performs the whole step of updatings
    void run_one();

    // SAMPLING (UPDATE) METHODS
    // REP-PP BLOCK
    // sample non-allocated jumps
    void sample_jumps_na();
    // sample allocated jumps
    void sample_jumps_a();
    // sample non-allocated means
    void sample_means_na();
    // sample allocated means
    void sample_means_a();
    // sample non-allocated deltas
    void sample_deltas_na();
    // sample allocated deltas
    void sample_deltas_a();
    // sample cluster allocations and relabel the parameters
    void sample_allocations_and_relabel();
    // sample u
    inline void sample_u(){
      double T = a_jumps.sum() + na_jumps.sum();
      u = gamma_rng(ndata, T, Rng::Instance().get());
    };

    // ETAS: virtual because for dim_fact=1, we directly invert scalars, not resolving systems!
    virtual void sample_etas()=0;

    // SIGMA_BAR: could be virtual because for dim_fact=1, we can exploit that eta^T eta is scalar..
    void sample_sigma_bar();

    // LAMBDA BLOCK : identical for both uni/multi factor since uni uses a matrixXd with 1 column.
    void sample_Psi();
    void sample_tau();
    void sample_Phi();
    // virtual: in uni cond process does not depend on Lambda
    virtual void sample_Lambda() = 0;

    // will be private
    inline double laplace(double u) const {
        return std::pow(_beta_jump, _alpha_jump) / std::pow(_beta_jump + u, _alpha_jump);
    }

    //will be private:  for update of Lambda! Used in sample_lambda().
    inline double compute_exp_lik(const MatrixXd& lamb) const;
    inline double compute_exp_prior(const MatrixXd& lamb) const;

    virtual void get_state_as_proto(google::protobuf::Message *out_) = 0;

    void print_debug_string();


    void _relabel();

    void set_verbose() { verbose = !verbose; }

    virtual double lpdf_given_clus(
        const VectorXd &x, const VectorXd &mu, const prec_t &sigma)  = 0;

    virtual double lpdf_given_clus_multi(
        const std::vector<fact_t> &x, const VectorXd &mu,
        const prec_t &sigma) = 0;

    inline void set_dim_factor() {
      dim_fact = params.get_dimf();
    }

    virtual VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) = 0;

    double mean_acceptance_rate() {
        return (1.0 * acc_mean) / (1.0 * tot_mean);
    }

    virtual void print_data_by_clus(int clus) = 0;
};


class MultivariateConditionalMCMC: public ConditionalMCMC<
    BaseMultiPrec, PrecMat, VectorXd> {

public:
    MultivariateConditionalMCMC() {}

    MultivariateConditionalMCMC(BaseDeterminantalPP *pp_mix, BasePrec *g,
                                const Params &params);

    // initializes the etas, projecting the data onto Col(Lambda):
    // it is for both uni/multi factor cases, but implemented differently because of the least square systems.
    void initialize_etas(const MatrixXd &dat) override;

    // initialize 10 (or less) allocated means: if <10 data, take the means on
    // the data, otherwise choose the means on randomly selected data
    void initialize_allocated_means() override;


    //ETAS
    void sample_etas() override;
    // LAMBDA
    void sample_Lambda() override;

    void get_state_as_proto(google::protobuf::Message *out_) override;

    double lpdf_given_clus(
        const VectorXd &x, const VectorXd &mu, const PrecMat &sigma)
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    double lpdf_given_clus_multi(
        const std::vector<VectorXd> &x, const VectorXd &mu, const PrecMat &sigma) override
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) override;

    void print_data_by_clus(int clus);

};

class UnivariateConditionalMCMC : public ConditionalMCMC<
    BaseUnivPrec, double, double>
{
public:
    UnivariateConditionalMCMC() {}

    UnivariateConditionalMCMC(BaseDeterminantalPP *pp_mix, BasePrec *g,
                              const Params &params);

    // initializes the etas, projecting the data onto Col(Lambda):
    // it is for both uni/multi factor cases, but implemented differently because of the least square systems.
    void initialize_etas(const MatrixXd &dat) override;

    void initialize_allocated_means() override;

    //ETAS
    void sample_etas() override;
    // LAMBDA
    void sample_Lambda() override;

    void get_state_as_proto(google::protobuf::Message *out_) override;

    double lpdf_given_clus(
        const VectorXd &x, const VectorXd &mu, const double &sigma)
    {
        return stan::math::normal_lpdf(x(0), mu(0), 1.0 / sqrt(sigma));
    }

    double lpdf_given_clus_multi(
        const std::vector<double> &x, const VectorXd &mu, const double &sigma) override
    {
        return stan::math::normal_lpdf(x, mu(0), 1.0 / sqrt(sigma));
    }


    VectorXd compute_grad_for_clus(int clus, const VectorXd& mean) override;

    void print_data_by_clus(int clus);
};


#include "conditional_mcmc_imp.hpp"

#endif
