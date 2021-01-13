#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <fstream>
#include <iomanip>
#include <stan/math/prim.hpp>

#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../precs/delta_gamma.hpp"
#include "../../precs/delta_wishart.hpp"
#include "../../../protos/cpp/params.pb.h"
#include "../../point_process/multi_factor_dpp.hpp"
#include "../../point_process/uni_factor_dpp.hpp"
using namespace Eigen;

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

    std::cout<<"data: "<<data<<std::endl;

    Eigen::MatrixXd ranges(2, d);
    ranges.row(0) = RowVectorXd::Constant(d, -50);
    ranges.row(1) = RowVectorXd::Constant(d, 50);

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // just a (dir param) and proposal for Lambda in MH step count, besides pp_mix params
    std::cout<<"params.a (Dir) = "<<params.a()<<std::endl;

    double prop_m_sigma = 1; // Useless in this test
    std::vector<double> prop_sigmas{0.01,0.1,1};
    int log_every = 1000;
    int niter = 50000;
    ofstream myfile;

    // Only Lambda block is free; others are passed fixed.
    // Also pass parameters of proposal for tuning (no more in params.proto for testing)
    // pp_mix and g are passed even if not used because the quantities are fixed.
    for (auto p_sigma : prop_sigmas){
    BaseDeterminantalPP *pp_mix = make_dpp(params, ranges);

    BasePrec *g = make_delta(params);

    MultivariateConditionalMCMC sampler(pp_mix, g, params, Mus, sigma_bar, Etas, p_sigma , prop_m_sigma);

    myfile.open("./src/spikes/inference_test/test_LambdaBlock_p4_d2_3g_inf.txt", ios::app);
    myfile <<"LambdaBlock hyperparameter: a (dir) = "<<params.a()<<"\n";
    myfile << "Proposal Lambda sigma = "<<p_sigma<<"\n";

    sampler.initialize(data);

    myfile <<"Initialization: \n"<<sampler.Lambda<<"\n";

    // assume burnin = niter
    for (int i = 0; i < niter; i++) {
      sampler.run_one();
      if ((i + 1) % log_every == 0) {
        std::cout<<"Burnin, iter #"<< i + 1<< " / "<< niter<<std::endl;
      }
    }

    MatrixXd average_Lambda = MatrixXd::Zero(p,d);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        average_Lambda += sampler.Lambda;
        if (i % 1000 == 0){
             std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    average_Lambda /= niter;
    double acc_rate = sampler.Lambda_acceptance_rate();

    std::cout<<"average_Lambda= "<<average_Lambda<<std::endl;

    myfile << "Lambda acceptance rate = "<<std::setprecision(5)<<acc_rate<<"\n";
    myfile << "Accepted sampled Lambda = "<<sampler.acc_sampled_Lambda<<"\n";
    myfile << "Average Lambda after "<<niter<<" iterations: \n";
    myfile << average_Lambda << "\n";
    myfile.close();

    }



    return 0;
}
