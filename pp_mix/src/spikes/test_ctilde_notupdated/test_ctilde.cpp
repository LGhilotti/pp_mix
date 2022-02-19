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



MatrixXd generate_etas(const MatrixXd& mus, const std::vector<MatrixXd>& deltas,
                        const VectorXi& c_alloc) {

    MatrixXd out (c_alloc.size(),mus.cols());

    for (int i = 0; i < c_alloc.size(); i++) {
      out.row(i) = stan::math::multi_normal_prec_rng(mus.row(c_alloc(i)), deltas[c_alloc(i)], Rng::Instance().get());
    }

    return out;
}


MatrixXd generate_data(const MatrixXd& lambda, const MatrixXd& etas, const VectorXd& sigma_bar) {

  MatrixXd out(etas.rows(),lambda.rows());
  MatrixXd means = lambda*etas.transpose();
  MatrixXd sigma_bar_mat = sigma_bar.asDiagonal();
  for (int i = 0; i < etas.rows(); i ++ ) {
    out.row(i) = stan::math::multi_normal_prec_rng(means.col(i),sigma_bar_mat, Rng::Instance().get());
  }

  return out;

}


int main() {

    // we consider p=4, d=2
    // we fix Lambda, M=3 (3 groups), mu1,mu2,mu3 Delta1, Delta2,Delta3 cluster labels, Sigmabar
    // and simulate etas and data.
    const int p = 4;
    int d = 2;
    MatrixXd Lambda(p,d);
    Lambda << 0, 1,
              1, 0,
              -1, 0,
              0, -1;

    const int M = 3;
    VectorXd mu_0(d), mu_1(d), mu_2(d);
    mu_0 << 5. , 0.;
    mu_1 << 0 , 10.;
    mu_2 << -5., -10. ;
    MatrixXd Mus(M,d);
    Mus.row(0)=mu_0;
    Mus.row(1)=mu_1;
    Mus.row(2)=mu_2;

    MatrixXd Delta_0(d,d), Delta_1(d,d), Delta_2(d,d);
    Delta_0 = MatrixXd::Identity(d,d);
    Delta_1 = MatrixXd::Identity(d,d);
    Delta_2 = MatrixXd::Identity(d,d);
    std::vector<MatrixXd> Deltas{Delta_0,Delta_1,Delta_2};

    VectorXd sigma_bar (VectorXd::Constant(p, 2.0));

    // we will consider 120 observations
    VectorXi cluster_alloc(30);
    cluster_alloc.head(10) = VectorXi::Zero(10);
    cluster_alloc.segment(10,10) = VectorXi::Ones(10);
    cluster_alloc.tail(10) = VectorXi::Constant(10, 2);

    MatrixXd Etas = generate_etas(Mus, Deltas, cluster_alloc);

    MatrixXd data = generate_data(Lambda, Etas, sigma_bar);

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use all params

    int log_every=1;
    int ntrick = 0;
    int burnin = 2;
    int niter=2;
    int thin = 1;


    DeterminantalPP* pp_mix = make_dpp(params);

    BasePrec* g = make_delta(params);

    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params);

    sampler.initialize(data);

    std::ofstream myfile;
    myfile.open("./src/spikes/test_ctilde_notupdated/test_c_notupdated.txt", std::ios::app);

    myfile<< "Initial Lambda: \n"<< sampler.get_Lambda()<<"\n";

/*
    for (int i = 0; i < ntrick; i++) {
        sampler.run_one_trick();
        if ((i + 1) % log_every == 0) {
            myfile<< "Trick, iter #"<< i + 1<< " / "<< ntrick<<"\n";
            //myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
            myfile<< "Grad_log_ad: \n"<< sampler.get_grad_log_ad()<<"\n";
            myfile<< "Grad_log_analytic: \n"<< sampler.get_grad_log_analytic()<<"\n";

        }
    }
*/


    for (int i = 0; i < burnin; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            myfile<<"Burnin, iter #"<< i + 1<< " / "<< burnin<<"\n";
            //myfile<< "Means_na: \n"<< sampler.get_na_means()<<"\n";
            myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
            //myfile<< "diff_log_dens_analytic: \n"<< sampler.get_ln_dens_analytic()<<"\n";

          }
    }

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            myfile<<"Running, iter #"<< i + 1<< " / "<< niter<<"\n";
            //myfile<< "Means_na: \n"<< sampler.get_na_means()<<"\n";
            myfile<< "Lambda: \n"<< sampler.get_Lambda()<<"\n";
            //myfile<< "diff_log_dens_analytic: \n"<< sampler.get_ln_dens_analytic()<<"\n";

          }
    }

    myfile.close();
    std::cout<<"acceptance lambda: "<<sampler.Lambda_acceptance_rate()<<std::endl;


    std::cout<<"END!"<<std::endl;
    return 0;


}
