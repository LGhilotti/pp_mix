//#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
//#include <stan/math/prim.hpp>
#include <stan/math/fwd.hpp>
#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>

#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../precs/delta_gamma.hpp"
#include "../../precs/delta_wishart.hpp"
#include "../../point_process/determinantalPP.hpp"

using namespace Eigen;
using namespace stan::math;


int main() {

    // we consider p=4, d=2
    // we fix Lambda, M=3 (3 groups), mu1,mu2,mu3 Delta1, Delta2,Delta3 cluster labels, Sigmabar
    // and simulate etas and data.
    const int p = 3;
    int d = 2;
    const int n = 5;
    MatrixXd data (n,p);
    for (int i=0; i<n; i++){
      for (int j=0; j<p; j++){
        data(i,j) = stan::math::bernoulli_rng(0.5,  Rng::Instance().get());
      }
    }

    Eigen::MatrixXd ranges(2, d);
    ranges.row(0) = RowVectorXd::Constant(d, -50);
    ranges.row(1) = RowVectorXd::Constant(d, 50);

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use all params

    int log_every=1;
    int ntrick = 1;
    int burnin = 2;
    int niter=2;
    int thin = 1;


    DeterminantalPP* pp_mix = make_dpp(params, ranges);

    BasePrec* g = make_delta(params, d);

    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params, d);

    sampler.initialize_binary(data);

    Eigen::VectorXi init_allocs_(n,1);
    init_allocs_ << 0,1,1,0,2;
    sampler.set_clus_alloc(init_allocs_);
    sampler._relabel();
    std::cout<<"Number means in trick phase: "<< sampler.get_num_a_means()<<std::endl;

    //std::ofstream myfile;
    //myfile.open("./src/spikes/test_derivatives/test_der.txt", std::ios::app);

    //myfile<< "Initial Lambda: \n"<< sampler.get_Lambda()<<"\n";


    for (int i = 0; i < ntrick; i++) {
        sampler.run_one_trick_binary();
        if ((i + 1) % log_every == 0) {
            //myfile<< "Trick, iter #"<< i + 1<< " / "<< ntrick<<"\n";
            //myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
            //myfile<< "Grad_log_ad: \n"<< sampler.get_grad_log_ad()<<"\n";
            //myfile<< "Grad_log_analytic: \n"<< sampler.get_grad_log_analytic()<<"\n";
            std::cout<<"trick, iter: "<<i+1<<std::endl;
        }
    }



    for (int i = 0; i < burnin; i++) {
        sampler.run_one_binary();
        if ((i + 1) % log_every == 0) {
          std::cout<<"burnin, iter: "<<i+1<<std::endl;

        }
    }

    for (int i = 0; i < niter; i++) {
        sampler.run_one_binary();
        if ((i + 1) % log_every == 0) {
          std::cout<<"running, iter: "<<i+1<<std::endl;

          }
    }

    //myfile.close();
    //std::cout<<"acceptance lambda: "<<sampler.Lambda_acceptance_rate()<<std::endl;


    std::cout<<"END!"<<std::endl;
    return 0;


}
