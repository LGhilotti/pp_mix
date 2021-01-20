#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <stan/math/prim.hpp>

#include "../../conditional_mcmc.hpp"
#include "../../tmp_conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../precs/delta_gamma.hpp"
#include "../../precs/delta_wishart.hpp"
#include "../../point_process/multi_factor_dpp.hpp"
#include "../../point_process/uni_factor_dpp.hpp"

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

  // we consider p=50, d=2
  // we fix Lambda, M=3 (2 groups), mu1,mu2,mu3 Delta1, Delta2,Delta3 cluster labels, Sigmabar
  // and simulate etas and data.
  const int p = 50;
  const int d = 2;
  MatrixXd Lambda = MatrixXd::Zero(p,d);
  Lambda.block(0,0,25,1) = VectorXd::Ones(25);
  Lambda.block(25,1,25,1) = - VectorXd::Ones(25);

  const int M = 3;
  VectorXd mu_0(d), mu_1(d), mu_2(d);
  mu_0 << -5. , 10. ;
  mu_1 << 0. , -5. ;
  mu_2 << 5. , -5. ;
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
  VectorXi cluster_alloc(120);
  cluster_alloc.head(40) = VectorXi::Zero(40);
  cluster_alloc.segment(40,40) = VectorXi::Ones(40);
  cluster_alloc.tail(40) = VectorXi::Constant(40, 2);

    MatrixXd Etas = generate_etas(Mus, Deltas, cluster_alloc);

    MatrixXd data = generate_data(Lambda, Etas, sigma_bar);


    Eigen::MatrixXd ranges(2, d);
    ranges.row(0) = RowVectorXd::Constant(d, -50);
    ranges.row(1) = RowVectorXd::Constant(d, 50);

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use all params

    std::vector<double> prop_mean_sigmas{0.01};
    std::vector<double> prop_lambda_sigmas{0.001,0.01,0.1};

    // assume burnin = niter
    int log_every = 1000;
    int niter = 50000; // GOOD: with lots of iters we converge to 2 clusters, while with 1000 iter only to 3-4 clusters.
    int ntrick= 15000;
    std::ofstream myfile_base;
    std::ofstream myfile_trick;

    myfile_base.open("./src/spikes/p50_inference_test/test_all_p50_d2_3g_compare.txt", std::ios::app);

    myfile_base <<"Original parameters: \n"<<"number of clusters: "<<M<<"\n";
    myfile_base <<"allocated means: \n"<<Mus<<"\n";
    myfile_base <<"allocated deltas: \n"<<Deltas[0]<<"\n"<<Deltas[1]<<"\n";
    myfile_base <<"cluster allocations: \n"<<cluster_alloc<<"\n";
    myfile_base <<"original etas: \n"<<Etas<<"\n";
    myfile_base <<"original sigma bar: "<<sigma_bar<<"\n";
    myfile_base <<"original Lambda: \n"<<Lambda<<"\n";
    myfile_base.close();


    myfile_trick.open("./src/spikes/p50_inference_test/test_all_p50_d2_3g_trick_compare.txt", std::ios::app);

    myfile_trick <<"Original parameters: \n"<<"number of clusters: "<<M<<"\n";
    myfile_trick <<"allocated means: \n"<<Mus<<"\n";
    myfile_trick <<"allocated deltas: \n"<<Deltas[0]<<"\n"<<Deltas[1]<<"\n";
    myfile_trick <<"cluster allocations: \n"<<cluster_alloc<<"\n";
    myfile_trick <<"original etas: \n"<<Etas<<"\n";
    myfile_trick <<"original sigma bar: "<<sigma_bar<<"\n";
    myfile_trick <<"original Lambda: \n"<<Lambda<<"\n";
    myfile_trick.close();

    // I want to make the program estimate this quantity
    /*
    std::vector<std::vector<int>> obs_by_clus(2);
    obs_by_clus[0].resize(50);
    obs_by_clus[1].resize(50);
    std::iota(std::begin(obs_by_clus[0]), std::end(obs_by_clus[0]), 0);
    std::iota(std::begin(obs_by_clus[1]), std::end(obs_by_clus[1]), 50);
    */
    for (auto prop_m_sigma : prop_mean_sigmas){
      for (auto prop_l_sigma : prop_lambda_sigmas){
    BaseDeterminantalPP *pp_mix = make_dpp(params, ranges);

    BasePrec *g = make_delta(params);

    Test::MultivariateConditionalMCMC sampler(pp_mix, g, params, prop_m_sigma, prop_l_sigma);



    sampler.initialize(data);

    myfile_trick.open("./src/spikes/p50_inference_test/test_all_p50_d2_3g_trick_compare.txt", std::ios::app);
/*
    myfile << "Initialization! Parameters: \n"<<"Initial tau: "<<sampler.tau<<"\n";
    myfile << "Initial Phi: \n"<<sampler.Phi<<"\n"<<"Initial Psi: \n"<<sampler.Psi<<"\n";
    myfile << "Initial Lambda: \n"<<sampler.Lambda<<"\n";
    myfile <<"Initial Sigma bar: \n"<<sampler.sigma_bar<<"\n";
    myfile <<"Initial etas: \n"<<sampler.etas<<"\n";
    myfile <<"Initial number of clusters: "<<sampler.a_means.rows()<<"\n";
    myfile <<"Initial allocated means: \n"<<sampler.a_means<<"\n";
    myfile <<"Initial allocated deltas: \n";
    for (int i = 0; i < sampler.a_means.rows(); i++){
      myfile <<sampler.a_deltas[i]<<"\n";
    }
    myfile <<"Initial allocated jumps: "<<sampler.a_jumps<<"\n";
    myfile <<"Initial non allocated means: \n"<<sampler.na_means<<"\n";
    myfile <<"Initial non allocated deltas: \n";
    for (int i = 0; i < sampler.na_means.rows(); i++){
      myfile <<sampler.na_deltas[i]<<"\n";
    }
    myfile <<"Initial non allocated jumps: "<<sampler.na_jumps<<"\n";

*/

    std::cout<<"Initialization ALL GOOD!"<<std::endl;

    for (int i = 0; i < ntrick; i++) {
      sampler.run_one_trick();
      if ((i + 1) % log_every == 0) {
        std::cout<<"Trick, iter #"<< i + 1<< " / "<< ntrick<<std::endl;
      }
    }
    myfile_trick << "Proposal means sigma: "<<prop_m_sigma<<"\n";
    myfile_trick << "Proposal lambda sigma: "<<prop_l_sigma<<"\n";
    myfile_trick << "##### After settlement phase (current state) \n";
    myfile_trick << "#### Means Acceptance rate: "<<sampler.a_means_acceptance_rate()<<"\n";
    myfile_trick << "#### Lambda Acceptance rate: "<<sampler.Lambda_acceptance_rate()<<"\n";
    myfile_trick << "Number Allocated means: "<<sampler.a_means.rows()<<"\n";
    myfile_trick <<"cluster allocations: \n"<<sampler.clus_alloc<<"\n";
    myfile_trick <<"Lambda: \n"<<sampler.Lambda<<"\n";
    myfile_trick <<"allocated means: \n"<<sampler.a_means<<"\n";
    myfile_trick << "current data means (Lambda*etas): \n"<<sampler.Lambda * sampler.etas.transpose() <<"\n";

    for (int i = 0; i < niter; i++) {
      sampler.run_one();
      if ((i + 1) % log_every == 0) {
        std::cout<<"Burnin, iter #"<< i + 1<< " / "<< niter<<std::endl;
      }
    }

    double average_nclus = 0.;
    VectorXd average_sigma_bar = VectorXd::Zero(p);
    MatrixXd average_Lambda = MatrixXd::Zero(p,d);
    MatrixXd average_etas = MatrixXd::Zero(data.rows(),d);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        average_nclus += sampler.a_means.rows();
        average_sigma_bar += sampler.sigma_bar;
        average_Lambda += sampler.Lambda;
        average_etas += sampler.etas;
        if ((i+1) % log_every == 0){
             std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    average_nclus /= niter;
    average_sigma_bar /= niter;
    average_Lambda /= niter;
    average_etas /= niter;


    myfile_trick << "#### Means Acceptance rate: "<<sampler.a_means_acceptance_rate()<<"\n";
    myfile_trick << "#### Lambda Acceptance rate: "<<sampler.Lambda_acceptance_rate()<<"\n";
    myfile_trick << "Average nclus after "<<niter<<" iterations: \n";
    myfile_trick << average_nclus << "\n";
    myfile_trick << "Average sigma bar: \n"<<average_sigma_bar<<"\n";
    myfile_trick << "Average Lambda: \n"<<average_Lambda<<"\n";
    myfile_trick << "Average etas: \n"<<average_etas<<"\n";
    myfile_trick << "Final iteration parameters: \n"<<"number of clusters: "<<sampler.a_means.rows()<<"\n";
    myfile_trick <<"allocated means: \n"<<sampler.a_means<<"\n";
    myfile_trick <<"allocated deltas: \n";
    for (int i = 0; i < sampler.a_means.rows(); i++){
      myfile_trick <<sampler.a_deltas[i]<<"\n";
    }
    myfile_trick <<"cluster allocations: \n"<<sampler.clus_alloc<<"\n";
    myfile_trick.close();

  }
}


//////////// BASE (NON trick)
for (auto prop_m_sigma : prop_mean_sigmas){
  for (auto prop_l_sigma : prop_lambda_sigmas){
    BaseDeterminantalPP *pp_mix = make_dpp(params, ranges);

    BasePrec *g = make_delta(params);

    MultivariateConditionalMCMC sampler(pp_mix, g, params, prop_m_sigma, prop_l_sigma);



    sampler.initialize(data);

    myfile_base.open("./src/spikes/p50_inference_test/test_all_p50_d2_3g_compare.txt", std::ios::app);
    /*
    myfile << "Initialization! Parameters: \n"<<"Initial tau: "<<sampler.tau<<"\n";
    myfile << "Initial Phi: \n"<<sampler.Phi<<"\n"<<"Initial Psi: \n"<<sampler.Psi<<"\n";
    myfile << "Initial Lambda: \n"<<sampler.Lambda<<"\n";
    myfile <<"Initial Sigma bar: \n"<<sampler.sigma_bar<<"\n";
    myfile <<"Initial etas: \n"<<sampler.etas<<"\n";
    myfile <<"Initial number of clusters: "<<sampler.a_means.rows()<<"\n";
    myfile <<"Initial allocated means: \n"<<sampler.a_means<<"\n";
    myfile <<"Initial allocated deltas: \n";
    for (int i = 0; i < sampler.a_means.rows(); i++){
      myfile <<sampler.a_deltas[i]<<"\n";
    }
    myfile <<"Initial allocated jumps: "<<sampler.a_jumps<<"\n";
    myfile <<"Initial non allocated means: \n"<<sampler.na_means<<"\n";
    myfile <<"Initial non allocated deltas: \n";
    for (int i = 0; i < sampler.na_means.rows(); i++){
      myfile <<sampler.na_deltas[i]<<"\n";
    }
    myfile <<"Initial non allocated jumps: "<<sampler.na_jumps<<"\n";

    */

    std::cout<<"Initialization ALL GOOD!"<<std::endl;

    for (int i = 0; i < niter+ntrick; i++) {
      sampler.run_one();
      if ((i + 1) % log_every == 0) {
        std::cout<<"Burnin, iter #"<< i + 1<< " / "<< niter<<std::endl;
      }
    }

    double average_nclus = 0.;
    VectorXd average_sigma_bar = VectorXd::Zero(p);
    MatrixXd average_Lambda = MatrixXd::Zero(p,d);
    MatrixXd average_etas = MatrixXd::Zero(data.rows(),d);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        average_nclus += sampler.a_means.rows();
        average_sigma_bar += sampler.sigma_bar;
        average_Lambda += sampler.Lambda;
        average_etas += sampler.etas;
        if ((i+1) % log_every == 0){
             std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    average_nclus /= niter;
    average_sigma_bar /= niter;
    average_Lambda /= niter;
    average_etas /= niter;


    myfile_base << "#### Means Acceptance rate: "<<sampler.a_means_acceptance_rate()<<"\n";
    myfile_base << "#### Lambda Acceptance rate: "<<sampler.Lambda_acceptance_rate()<<"\n";
    myfile_base << "Average nclus after "<<niter<<" iterations: \n";
    myfile_base << average_nclus << "\n";
    myfile_base << "Average sigma bar: \n"<<average_sigma_bar<<"\n";
    myfile_base << "Final iteration parameters: \n"<<"number of clusters: "<<sampler.a_means.rows()<<"\n";
    myfile_base <<"allocated means: \n"<<sampler.a_means<<"\n";
    myfile_base <<"allocated deltas: \n";
    for (int i = 0; i < sampler.a_means.rows(); i++){
      myfile_base <<sampler.a_deltas[i]<<"\n";
    }
    myfile_base <<"cluster allocations: \n"<<sampler.clus_alloc<<"\n";
    myfile_base.close();

}
}

    return 0;
}
