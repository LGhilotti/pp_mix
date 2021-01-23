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

#include "../../mala_conditional_mcmc.hpp"
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
    const int d = 2;
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

    double prop_m_sigma = 0.1;
    //double prop_l_sigma = 0.01;
    std::vector<double> mala_params {0.0001,0.001,0.01};

    int log_every=100;
    int ntrick = 1000;
    int niter=10000;
    std::ofstream myfile;
    /*
    myfile.open("./src/spikes/MALA/test_mala_p4.txt", std::ios::app);
    myfile <<"Original parameters: \n"<<"number of clusters: "<<M<<"\n";
    myfile <<"allocated means: \n"<<Mus<<"\n";
    myfile <<"allocated deltas: \n"<<Deltas[0]<<"\n"<<Deltas[1]<<"\n";
    myfile <<"cluster allocations: \n"<<cluster_alloc<<"\n";
    myfile <<"original etas: \n"<<Etas<<"\n";
    myfile <<"original sigma bar: "<<sigma_bar<<"\n";
    myfile <<"original Lambda: \n"<<Lambda<<"\n";
    myfile.close();*/

    for (auto mala_p : mala_params){

      DeterminantalPP *pp_mix = make_dpp(params, ranges);

      BasePrec *g = make_delta(params);

      Mala::MalaMultiMCMC sampler(pp_mix, g, params, prop_m_sigma, mala_p);

      sampler.initialize(data);
/*
      myfile.open("./src/spikes/MALA/test_mala_p4.txt", std::ios::app);
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
      
      myfile.open("./src/spikes/MALA/test_mala_p4_afterMod_prova.txt", std::ios::app);

      myfile << "\nntrick = "<<ntrick<<"\n";
      myfile << "niter = "<<niter<<"\n";
      myfile << "Proposal means sigma: "<<prop_m_sigma<<"\n";
      myfile << "Proposal mala step: "<<mala_p<<"\n";
      myfile << "##### After settlement phase (current state) \n";
      myfile << "#### Means Acceptance rate: "<< std::fixed<<std::setprecision(5) <<sampler.a_means_acceptance_rate()<<"\n";
      myfile << "#### Lambda Acceptance rate: "<< std::fixed<<std::setprecision(5) <<sampler.Lambda_acceptance_rate()<<"\n";
      myfile << "Number Allocated means: "<<sampler.a_means.rows()<<"\n";
      myfile <<"cluster allocations: \n"<<sampler.clus_alloc<<"\n";
      myfile <<"Lambda: \n"<<sampler.Lambda<<"\n";
      myfile <<"allocated means: \n"<<sampler.a_means<<"\n";
      myfile << "current data means (Lambda*etas): \n"<<sampler.Lambda * sampler.etas.transpose() <<"\n";

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
              std::cout << "iter: " << i +1 << " / " << niter << std::endl;
          }
      }
      average_nclus /= niter;
      average_sigma_bar /= niter;
      average_Lambda /= niter;
      average_etas /= niter;


      myfile << "#### Means Acceptance rate: "<<std::fixed<<std::setprecision(5) <<sampler.a_means_acceptance_rate()<<"\n";
      myfile << "#### Lambda Acceptance rate: "<< std::fixed<<std::setprecision(5) <<sampler.Lambda_acceptance_rate()<<"\n";
      myfile << "Average nclus after "<<niter<<" iterations: \n";
      myfile << average_nclus << "\n";
      myfile << "Average sigma bar: \n"<<average_sigma_bar<<"\n";
      myfile << "Average Lambda: \n"<<average_Lambda<<"\n";
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
