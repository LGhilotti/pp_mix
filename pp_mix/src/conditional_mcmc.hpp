#ifndef CONDITIONAL_MCMC
#define CONDITIONAL_MCMC

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <stan/math/prim/mat.hpp>

#include "rng.hpp"
#include "point_process/base_pp.hpp"
#include "jumps/base_jump.hpp"
#include "covs/base_cov.hpp"
#include "covs/covmat.hpp"
#include "utils.hpp"
#include "../protos/cpp/state.pb.h"

using namespace Eigen;

class ConditionalMCMC {
 protected:
     int dim;
     int ndata;
     MatrixXd data;
     double prior_ratio, lik_ratio;
     std::vector<std::vector<VectorXd>> data_by_clus;

     // STATE
     int nclus;
     double u;
     VectorXi clus_alloc;
     VectorXd a_jumps, na_jumps;
     MatrixXd a_means, na_means;
     std::vector<PrecMat> a_precs, na_precs;

     // DISTRIBUTIONS
     BasePP *pp_mix;
     BaseJump *h;
     BasePrec *g;

     // FOR DEBUGGING
     double  birth_prob, arate;
     bool verbose = false;

 public:
    ConditionalMCMC() {}
    ~ConditionalMCMC() {
        delete pp_mix;
        delete h;
        delete g;
    }

    ConditionalMCMC(BasePP* pp_mix, BaseJump* h, BasePrec* g);

    void initialize(const MatrixXd& data);

    void run_one();

    void sample_allocations_and_relabel();

    void sample_means();

    void sample_vars();

    void sample_jumps();
    
    MixtureState get_state_as_proto();

    void print_debug_string();

    void _relabel();

    void set_verbose() { verbose = !verbose; }

};

#endif