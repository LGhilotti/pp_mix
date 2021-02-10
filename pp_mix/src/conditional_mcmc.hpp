#ifndef MULTI_CONDITIONAL_MCMC
#define MULTI_CONDITIONAL_MCMC

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <functional>
#include <cmath>


//include <stan/math/fwd.hpp>
//#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>

#include <google/protobuf/message.h>

#include "rng.hpp"
#include "point_process/determinantalPP.hpp"
#include "precs/base_prec.hpp"
#include "utils.hpp"
#include "../protos/cpp/params.pb.h"

using namespace Eigen;
using namespace stan::math;

namespace MCMCsampler {
    class BaseLambdaSampler ;
    class BaseMeansSampler ;
}


namespace MCMCsampler {

class MultivariateConditionalMCMC {
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

    // NOTE: in each iteration (run_one), when updating etas, this structure is no more correct,
    // since it is not updated. But this structure is no more used in following updates, so
    // it is ok. This will be updated at next iteration in _relabel and used correctly
    std::vector<std::vector<VectorXd>> etas_by_clus;

    /* STATE */
    // Rep-pp
    double u;
    VectorXi clus_alloc;
    VectorXd a_jumps, na_jumps;
    MatrixXd a_means, na_means;
    std::vector<PrecMat> a_deltas, na_deltas;
    // etas: n x d matrix
    MatrixXd etas;
   
    //Sigma_bar
    VectorXd sigma_bar;
    //Lambda-block
    double tau;
    MatrixXd Phi;
    MatrixXd Psi;
    MatrixXd Lambda;

    // Lambda sampling callable object
    BaseLambdaSampler* sample_lambda;
    // Allocated means sampling callable object
    BaseMeansSampler* sample_alloc_means;

   
    // FOR DEBUGGING
    bool verbose = false;

    Params params;


 public:
   
    // DISTRIBUTIONS
    DeterminantalPP *pp_mix;
    BaseMultiPrec *g;

    
    MultivariateConditionalMCMC() {}
    ~MultivariateConditionalMCMC()
    {
         delete pp_mix;
         delete g;
    }

    MultivariateConditionalMCMC(DeterminantalPP *pp_mix, BasePrec *g,
                                const Params &params);


    void set_pp_mix(DeterminantalPP* pp_mix) {this->pp_mix = pp_mix;}
    void set_prec(BaseMultiPrec* g) {this->g = g;}
    void set_params(const Params & p);

    // initializes some of the members (data, dim, ndata,..) and state of sampler
    // The constructor only initialize some other members (pointers) and params field
    void initialize(const MatrixXd& dat);

    // initializes the etas, projecting the data onto Col(Lambda):
    // it is for both uni/multi factor cases, but implemented differently because of the least square systems.
    void initialize_etas(const MatrixXd &dat);

    void initialize_allocated_means();

    std::vector<VectorXd> proj_inside();
    bool is_inside(const VectorXd & point);

    // it performs the whole step of updatings
    void run_one();
    void run_one_trick();

    // SAMPLING (UPDATE) METHODS
    // REP-PP BLOCK
    // sample non-allocated jumps
    void sample_jumps_na();
    // sample allocated jumps
    void sample_jumps_a();
    // sample non-allocated means
    void sample_means_na(double psi_u);
    //sample non-allocated means with trick, without changing number
    void sample_means_na_trick();
    // sample allocated means: OBJECT FOR MANAGE MALA AND MH
    //void sample_means_a();
    // sample non-allocated deltas
    void sample_deltas_na();
    // sample allocated deltas
    void sample_deltas_a();
    // sample cluster allocations and relabel the parameters
    void sample_allocations_and_relabel();

    void _relabel();

    // sample u
    inline void sample_u(){
      double T = a_jumps.sum() + na_jumps.sum();
      u = gamma_rng(ndata, T, Rng::Instance().get());
    };

    // ETAS: virtual because for dim_fact=1, we directly invert scalars, not resolving systems!
    void sample_etas();

    // SIGMA_BAR: could be virtual because for dim_fact=1, we can exploit that eta^T eta is scalar..
    void sample_sigma_bar();

    // LAMBDA BLOCK : identical for both uni/multi factor since uni uses a matrixXd with 1 column.
    void sample_Psi();
    void sample_tau();
    void sample_Phi();
    // virtual: in uni cond process does not depend on Lambda: OBJECT TO MANAGE MALA AND MH
    //virtual void sample_Lambda() = 0;

    // will be private
    inline double laplace(double u) const {
        return std::pow(_beta_jump, _alpha_jump) / std::pow(_beta_jump + u, _alpha_jump);
    }

    
    // to store the current state in proto format
    void get_state_as_proto(google::protobuf::Message *out_);

    // to just print the current state for debugging
    void print_debug_string();

    void set_verbose() { verbose = !verbose; }
    
    double lpdf_given_clus(const VectorXd &x, const VectorXd &mu, const PrecMat &sigma)
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }
    
    double lpdf_given_clus_multi(
        const std::vector<VectorXd> &x, const VectorXd &mu, const PrecMat &sigma) 
    {
        return o_multi_normal_prec_lpdf(x, mu, sigma);
    }

    //virtual VectorXd compute_grad_for_clus(int clus, const VectorXd &mean) = 0;

    inline double a_means_acceptance_rate();

    inline double Lambda_acceptance_rate();

    void print_data_by_clus(int clus);

    int get_dim_data() const {return dim_data;}

    int get_dim_fact() const {return dim_fact;}

    double get_tau() const {return tau;}

    MatrixXd get_Psi() const {return Psi;}

    MatrixXd get_Phi() const {return Phi;}

    MatrixXd get_Lambda() const {return Lambda;}

    VectorXd get_sigma_bar() const {return sigma_bar;}

    MatrixXd get_data() const {return data;}

    MatrixXd get_etas() const {return etas;}

    int get_num_a_means() const {return a_means.rows();}

    int get_num_na_means() const {return na_means.rows();}

    RowVectorXd get_single_a_mean(int ind) const {return a_means.row(ind);}

    MatrixXd get_a_means_except_ind(int ind) const {return delete_row(a_means, ind);}

    MatrixXd get_a_means() const {return a_means;}

    MatrixXd get_na_means() const {return na_means;}

    MatrixXd& set_Lambda() {return Lambda;}

};



}


#endif
