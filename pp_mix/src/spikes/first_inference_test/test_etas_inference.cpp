#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
/*
#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/delimited_message_util.h>
*/
#include <stan/math/prim.hpp>
/*
#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../precs/delta_gamma.hpp"
#include "../../precs/delta_wishart.hpp"
#include "../../point_process/multi_factor_dpp.hpp"
#include "../../point_process/uni_factor_dpp.hpp"
*/
#include "../../rng.hpp"
//#include "../../../protos/cpp/params.pb.h"
using namespace Eigen;
using namespace stan::math;

class EtasSampler {
public:
  EtasSampler(const MatrixXd& lambda, const VectorXd& SigBar, const MatrixXd& mus, const std::vector<MatrixXd>& deltas,
              const std::vector<std::vector<int>>& obs_by_c, const VectorXi& cluster_alloc):
      Lambda(lambda), sigma_bar(SigBar), a_means(mus), a_deltas(deltas), obs_by_clus(obs_by_c), clus_alloc(cluster_alloc) {

    this->dim_fact = lambda.cols();

  }

  void initialize(const MatrixXd& dat){
    this->data = dat;
    ndata = data.rows();
    dim_data = data.cols();

    LLT<MatrixXd> M (Lambda.transpose() * Lambda);
    etas = (M.solve((data*Lambda).transpose())).transpose();

    return;
  }

  void run_one(){

    std::cout<<"sample etas"<<std::endl;
    MatrixXd M0(Lambda.transpose() * sigma_bar.asDiagonal());
    MatrixXd M1( M0 * Lambda);
    std::vector<MatrixXd> Sn_bar(a_means.rows());
    // type LLT for solving systems of equations
    std::vector<LLT<MatrixXd>> Sn_bar_cho (a_means.rows());

    for (int i=0; i < a_means.rows(); i++){
      Sn_bar[i]=M1+a_deltas[i];
      Sn_bar_cho[i]= LLT<MatrixXd>(Sn_bar[i]);
    }

    MatrixXd M2(M0*data.transpose());
    MatrixXd G(ndata,dim_fact);
    // known terms of systems is depending on the single data
    for (int i=0; i < a_means.rows(); i++){
      MatrixXd B(dim_fact,obs_by_clus[i].size());
      B = (a_deltas[i] * a_means.row(i).transpose()).replicate(1,B.cols());
      B +=M2(all,obs_by_clus[i]);
      // each modified row has solution for points in the cluster.
      G(obs_by_clus[i],all)=(Sn_bar_cho[i].solve(B)).transpose();
    }

    // Here, G contains (in each row) mean of full-cond, while precisions have to be taken from Sn_bar
    // Now, I sample each eta from the full-cond multi-normal
    for (int i=0; i < ndata; i++){
      etas.row(i)=multi_normal_prec_rng(G.row(i).transpose(), Sn_bar[clus_alloc(i)], Rng::Instance().get());
    }
    return;

  }

  // Etas
  MatrixXd etas;


private:
  int ndata, dim_data, dim_fact;
  MatrixXd data;
  MatrixXd Lambda;
  VectorXd sigma_bar;
  MatrixXd a_means;
  std::vector<MatrixXd> a_deltas;
  std::vector<std::vector<int>> obs_by_clus;
  VectorXi clus_alloc;

};
//end EtasSampler class

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
    // we fix Lambda, M=2 (2 groups), mu1,mu2, Delta1, Delta2, cluster labels, Sigmabar
    // and simulate etas and data.
    const int p = 4;
    const int d = 2;
    MatrixXd Lambda(p,d);
    Lambda << 0, 1,
              1, 0,
              -1, 0,
              0, -1;

    const int M = 2;
    VectorXd mu_0(d), mu_1(d);
    mu_0 << 10. , 0.;
    mu_1 << 0. , 10.;
    MatrixXd Mus(2,2);
    Mus.row(0)=mu_0;
    Mus.row(1)=mu_1;

    MatrixXd Delta_0(d,d), Delta_1(d,d);
    Delta_0 = MatrixXd::Identity(d,d);
    Delta_1 = MatrixXd::Identity(d,d);
    std::vector<MatrixXd> Deltas{Delta_0,Delta_1};

    VectorXd sigma_bar (VectorXd::Constant(p, 2.0));

    // we will consider 100 observations, first 50 from group 0, last 50 from group 1.
    VectorXi cluster_alloc(100);
    cluster_alloc.head(50) = VectorXi::Zero(50);
    cluster_alloc.tail(50) = VectorXi::Ones(50);

    MatrixXd Etas = generate_etas(Mus, Deltas, cluster_alloc);

    MatrixXd data = generate_data(Lambda, Etas, sigma_bar);

    std::cout<<"data: "<<data<<std::endl;

    Eigen::MatrixXd ranges(2, 2);
    ranges.row(0) = RowVectorXd::Constant(2, -50);
    ranges.row(1) = RowVectorXd::Constant(2, 50);

    std::vector<std::vector<int>> obs_by_clus(2);
    obs_by_clus[0].resize(50);
    obs_by_clus[1].resize(50);
    std::iota(std::begin(obs_by_clus[0]), std::end(obs_by_clus[0]), 0);
    std::iota(std::begin(obs_by_clus[1]), std::end(obs_by_clus[1]), 50);
    // DEBUG
    /*for (int j = 0; j < 2; j++){
      std::cout<<"observation cluster "<<j<<std::endl;
      for (int i = 0; i < 50; i++){
          std::cout<<obs_by_clus[j][i]<<std::endl;
      }
    }*/

    // Instead of using directly MultivariateConditionalMCMC and commenting out rows,
    // I just define a simple class which performs the same functionality
    EtasSampler sampler(Lambda, sigma_bar, Mus, Deltas, obs_by_clus, cluster_alloc );
    /*
    // USELESS IN THIS TEST
    BaseDeterminantalPP *pp_mix = make_dpp(params, ranges);


    BasePrec *g = make_delta(params);

    // this are not used because never do steps of MH
    double prop_l_sigma = 1;
    double prop_m_sigma = 1;

    // Only Sigma_bar is free; others are passed fixed.
    // Also pass parameters of proposal for tuning (no more in params.proto for testing)
    // pp_mix and g are passed even if not used because the quantity are fixed.
    MultivariateConditionalMCMC sampler(pp_mix, g, params, Lambda, Etas, prop_l_sigma, prop_m_sigma);
*/
    std::ofstream myfile;
    myfile.open("./src/spikes/inference_test/test_etas_inf.txt", std::ios::app);
    myfile <<"Original etas: \n"<<Etas<<"\n";

    sampler.initialize(data);

    myfile << "Initialization: \n"<<sampler.etas<<"\n";

    std::cout<<"Initialization ALL GOOD!"<<std::endl;


    // assume burnin = niter
    int log_every = 1000;
    int niter = 50000;
    for (int i = 0; i < niter; i++) {
      sampler.run_one();
      if ((i + 1) % log_every == 0) {
        std::cout<<"Burnin, iter #"<< i + 1<< " / "<< niter<<std::endl;
      }
    }

    MatrixXd average_etas = MatrixXd::Zero(data.rows(),d);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        average_etas += sampler.etas;
        if (i % 1000 == 0){
             std::cout<<"etas = "<<sampler.etas<<std::endl;
             std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    average_etas /= niter;

    std::cout<<"average_etas= "<<average_etas<<std::endl;

    myfile << "Average etas after "<<niter<<" iterations: \n";
    myfile << average_etas << "\n";
    myfile.close();

    return 0;
}
