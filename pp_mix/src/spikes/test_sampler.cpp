#include <Eigen/Dense>
#include <deque>
#include <string>
#include <memory>

#include <stan/math/prim/mat.hpp>

#include "../conditional_mcmc.hpp"
#include "../factory.hpp"
#include "../../protos/cpp/state.pb.h"
#include "../../protos/cpp/params.pb.h"


int main() {
    int data_per_clus = 50;
    Eigen::MatrixXd data = Eigen::MatrixXd(data_per_clus * 2, 2);
    Eigen::VectorXd mean1(2);
    mean1 << 3.0, 3.0;

    Eigen::VectorXd mean2(2);
    mean2 << -3.0, -3.0;

    double sigma = 0.3;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(2, 2) * sigma;

    for (int i = 0; i < data_per_clus; i++)
    {
        data.row(i) = stan::math::multi_normal_rng(
            mean1, cov, Rng::Instance().get());
    }

    for (int i = data_per_clus; i < 2 * data_per_clus; i++)
    {
        data.row(i) = stan::math::multi_normal_rng(
            mean2, cov, Rng::Instance().get());
    }

    Eigen::MatrixXd ranges(2, data.cols());
    ranges.row(0) = data.colwise().minCoeff();
    ranges.row(1) = data.colwise().maxCoeff();
    ranges *= 2;
    std::cout << "ranges: \n" << ranges << std::endl;

    std::string params_file = \
        "/home/mario/PhD/finiteDPP/pp_mix/pp_mix/resources/sampler_params.asciipb";
    Params params = loadTextProto<Params>(params_file);

    BasePP *pp_mix = make_pp(params);
    BaseJump *h = make_jump(params);
    BasePrec *g = make_prec(params);
    pp_mix->set_ranges(ranges);

    ConditionalMCMC sampler(pp_mix, h, g);
    sampler.initialize(data);

    std::deque<MixtureState> out;
    
    for (int i = 0; i < 10000; i++)
        sampler.run_one();

    // sampler.set_verbose();
    // int niter = 50000;
    // for (int i = 0; i < niter; i++) {
    //     sampler.run_one();

    //     std::cout << "press key";
    //     std::cin.get();

    //     out.push_back(sampler.get_state_as_proto());
    //     if (i % 1000 == 0) {
    //         std::cout << "iter: " << i << " / " << niter << std::endl;
    //     }
    // }
    sampler.print_debug_string();
}