#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <google/protobuf/message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/delimited_message_util.h>

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
#include "../../../protos/cpp/params.pb.h"
using namespace Eigen;
using namespace stan::math;

class SigmaSampler {
public:
  SigmaSampler(const Params& params, const MatrixXd& lambda, const MatrixXd& etas): Lambda(lambda), etas(etas) {
    set_params(params);
  }

  void initialize(const MatrixXd& dat){
    this->data = dat;
    ndata = data.rows();
    dim_data = data.cols();
    sigma_bar = _a_gamma/_b_gamma * VectorXd::Ones(dim_data);
    return;
  }

  void run_one(){

    VectorXd betas = VectorXd::Constant(dim_data, _b_gamma);
    betas+=0.5*data.colwise().squaredNorm().transpose() + 0.5*(etas*Lambda.transpose()).colwise().squaredNorm().transpose();
    for (int j=0; j < dim_data; j++){
      betas(j)-= Lambda.row(j)*etas.transpose()*data.col(j);
    }

    sigma_bar = Map<VectorXd>( gamma_rng(std::vector<double>(dim_data, ndata/2.0 + _a_gamma),
              std::vector<double>(betas.data(),betas.data()+betas.size()), Rng::Instance().get()).data(), dim_data);

    return;

  }

  //Sigma_bar
  VectorXd sigma_bar;

private:
  int ndata, dim_data, dim_fact;
  double _a_gamma, _b_gamma;
  MatrixXd data;
  Params params;
  MatrixXd Lambda;
  MatrixXd etas;

  void set_params(const Params& p) {
    this->params = p;
    this->dim_fact = params.dimf();
    this->_a_gamma = params.agamma();
    this->_b_gamma = params.bgamma();

    return;
  }

};
//end SigmaSampler class

MatrixXd generate_etas(const std::vector<VectorXd>& mus, const std::vector<MatrixXd>& deltas,
                        const VectorXi& c_alloc) {

    MatrixXd out (c_alloc.size(),mus[0].size());

    for (int i = 0; i < c_alloc.size(); i++) {
      out.row(i) = stan::math::multi_normal_prec_rng(mus[c_alloc(i)], deltas[c_alloc(i)], Rng::Instance().get());
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

template <typename T>
T loadTextProto(std::string filename) {
  std::ifstream ifs(filename);
  google::protobuf::io::IstreamInputStream iis(&ifs);
  T out;
  auto success = google::protobuf::TextFormat::Parse(&iis, &out);
  if (!success)
    std::cout << "An error occurred in 'loadTextProto'; success: " << success
              << std::endl;
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
    std::vector<VectorXd> Mus{mu_0,mu_1};

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

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);
    // the other parameters in Params are useless in this test
    std::cout<<"params.a_gamma = "<<params.agamma()<<std::endl;
    std::cout<<"params.b_gamma = "<<params.bgamma()<<std::endl;

    // Instead of using directly MultivariateConditionalMCMC and commenting out rows,
    // I just define a simple class which performs the same functionality
    SigmaSampler sampler(params, Lambda, Etas);
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
    myfile.open("./src/spikes/inference_test/test_sigmabar_inf.txt", std::ios::app);
    myfile <<"Sigma bar hyperparameters: a_gamma = "<<params.agamma()<<" ; b_gamma = "<<params.bgamma()<<"\n";

    sampler.initialize(data);

    myfile << "Initialization: \n"<<sampler.sigma_bar<<"\n";
    
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

    VectorXd average_sigma_bar = VectorXd::Zero(p);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        average_sigma_bar += sampler.sigma_bar;
        if (i % 1000 == 0){
             std::cout<<"sigma bar = "<<sampler.sigma_bar<<std::endl;
             std::cout << "iter: " << i << " / " << niter << std::endl;
        }
    }
    average_sigma_bar /= niter;

    std::cout<<"average_sigma_bar= "<<average_sigma_bar<<std::endl;

    myfile << "Average sigma bar after "<<niter<<" iterations: \n";
    myfile << average_sigma_bar << "\n";
    myfile.close();

    return 0;
}
