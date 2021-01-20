#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <stan/math/prim.hpp>

#include "../../../tmp_conditional_mcmc.hpp"
#include "../../../factory.hpp"
#include "../../../precs/delta_gamma.hpp"
#include "../../../precs/delta_wishart.hpp"
#include "../../../point_process/multi_factor_dpp.hpp"
#include "../../../point_process/uni_factor_dpp.hpp"

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

    // we consider p=200, d=2
    // we fix Lambda, M=3 (2 groups), mu1,mu2,mu3 Delta1, Delta2,Delta3 cluster labels, Sigmabar
    // and simulate etas and data.
    const int p = 200;
    const int d = 2;
    MatrixXd Lambda = MatrixXd::Zero(p,d);
    Lambda.block(0,0,100,1) = VectorXd::Ones(100);
    Lambda.block(100,1,100,1) = - VectorXd::Ones(100);

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

    // DATA DO NOT INFLUENCE THIS TEST, NOT IN THE POSTERIOR OF REP-PP BLOCK

    MatrixXd data = generate_data(Lambda, Etas, sigma_bar);


    Eigen::MatrixXd ranges(2, d);
    ranges.row(0) = RowVectorXd::Constant(d, -50);
    ranges.row(1) = RowVectorXd::Constant(d, 50);

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // NOTE: We use dpp, wishart, jump params

    std::vector<double> prop_sigmas{0.1};

    int log_every = 50;
    int niter = 1000;
    std::ofstream myfile;

    myfile.open("./src/spikes/inference_test/inf_multiple_params/test_etas_repblock_p200_inf.txt", std::ios::app);

    myfile <<"Original parameters: \n"<<"number of clusters: "<<M<<"\n";
    myfile <<"allocated means: \n"<<Mus<<"\n";
    myfile <<"allocated deltas: \n"<<Deltas[0]<<"\n"<<Deltas[1]<<"\n";
    myfile <<"cluster allocations: \n"<<cluster_alloc<<"\n";
    myfile <<"original etas: \n"<<Etas<<"\n";
    myfile.close();

    // I want to make the program estimate this quantity
    /*
    std::vector<std::vector<int>> obs_by_clus(2);
    obs_by_clus[0].resize(50);
    obs_by_clus[1].resize(50);
    std::iota(std::begin(obs_by_clus[0]), std::end(obs_by_clus[0]), 0);
    std::iota(std::begin(obs_by_clus[1]), std::end(obs_by_clus[1]), 50);
    */
    // I want to make the program estimate this quantity
    /*
    std::vector<std::vector<int>> obs_by_clus(2);
    obs_by_clus[0].resize(50);
    obs_by_clus[1].resize(50);
    std::iota(std::begin(obs_by_clus[0]), std::end(obs_by_clus[0]), 0);
    std::iota(std::begin(obs_by_clus[1]), std::end(obs_by_clus[1]), 50);
    */
    for (auto prop_m_sigma : prop_sigmas){
    BaseDeterminantalPP *pp_mix = make_dpp(params, ranges);

    BasePrec *g = make_delta(params);

    Test::MultivariateConditionalMCMC sampler(pp_mix, g, params, Lambda, sigma_bar, prop_m_sigma);

    sampler.initialize(data);

    myfile.open("./src/spikes/inference_test/inf_multiple_params/test_etas_repblock_p200_inf.txt", std::ios::app);
    myfile << "Initialization! Parameters: \n"<<"number of clusters: "<<sampler.a_means.rows()<<"\n";
    myfile <<"Initial allocated means: \n"<<sampler.a_means<<"\n";
    myfile <<"Initial allocated deltas: \n";
    for (int i = 0; i < sampler.a_means.rows(); i++){
      myfile <<sampler.a_deltas[i]<<"\n";
    }
    myfile <<"Initial etas: \n"<<sampler.etas<<"\n";

    std::cout<<"Initialization ALL GOOD!"<<std::endl;


    // assume burnin = niter

    for (int i = 0; i < niter; i++) {
      sampler.run_one();
      if ((i + 1) % log_every == 0) {
        std::cout<<"Burnin, iter #"<< i + 1<< " / "<< niter<<std::endl;
      }
    }

    double average_nclus = 0.;
    MatrixXd average_etas = MatrixXd::Zero(data.rows(),d);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        average_nclus += sampler.a_means.rows();
        average_etas += sampler.etas;
        if (i % 1000 == 0){
             std::cout<<"a_means = "<<sampler.a_means<<std::endl;
             std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    average_nclus /= niter;
    average_etas /= niter;

    std::cout<<"average_nclus= "<<average_nclus<<std::endl;

    myfile << "Proposal means sigma: "<<prop_m_sigma<<"\n";
    myfile << "#### Acceptance rate: "<<sampler.a_means_acceptance_rate()<<"\n";
    myfile << "Average nclus after "<<niter<<" iterations: \n";
    myfile << average_nclus << "\n";
    myfile << "Average etas: \n"<<average_etas<<"\n";
    myfile << "Final iteration parameters: \n"<<"number of clusters: "<<sampler.a_means.rows()<<"\n";
    myfile <<"allocated means: \n"<<sampler.a_means<<"\n";
    myfile <<"allocated deltas: \n";
    for (int i = 0; i < sampler.a_means.rows(); i++){
      myfile <<sampler.a_deltas[i]<<"\n";
    }
    myfile <<"cluster allocations: \n"<<sampler.clus_alloc<<"\n";
    myfile.close();
  }
  
    return 0;
}
