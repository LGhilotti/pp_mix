#include <deque>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
//#include <stan/math/prim.hpp>
//#include <stan/math/fwd.hpp>
//#include <stan/math/mix.hpp>
//#include <stan/math/prim.hpp>

#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "../../conditional_mcmc.hpp"
#include "../../factory.hpp"
#include "../../utils.hpp"
//#include "../../precs/delta_gamma.hpp"
//#include "../../precs/delta_wishart.hpp"
//#include "../../point_process/determinantalPP.hpp"
#include <Eigen/Dense>

using namespace Eigen;
using namespace stan::math;


template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

int main() {

      const int d=2;
      Eigen::MatrixXd ranges(2, d);
      ranges.row(0) = RowVectorXd::Constant(d, -50);
      ranges.row(1) = RowVectorXd::Constant(d, 50);

    MatrixXd datacsv = load_csv<MatrixXd>("/home/lorenzo/Documents/Tesi/github_repos/pp_mix/data/data_by_rule_gaussian/p_100_d_2_M_4_nperclus_50_data.csv");

    // We perform 1000 Mala steps for Lambda, other parameters fixed.
    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/valgrind/sampler_params_d2.asciipb";
    Params params = loadTextProto<Params>(params_file);

    int log_every=10;
    int ntrick = 1;
    int burnin = 1;
    int niter=100;
    int thin = 1;


    DeterminantalPP* pp_mix = make_dpp(params, ranges);

    BasePrec* g = make_delta(params);

    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params);

    sampler.initialize(datacsv);
    /*
    for (int i = 0; i < ntrick; i++) {
        sampler.run_one_trick();
        if ((i + 1) % log_every == 0) {
            std::cout<< "Trick, iter #"<< i + 1<< " / "<< ntrick<<std::endl;
        }
    }

    for (int i = 0; i < burnin; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            std::cout<<"Burnin, iter #"<< i + 1<< " / "<< burnin<<std::endl;
        }
    }
    */
    for (int i = 0; i < niter; i++) {
        sampler.run_one();

        if ((i + 1) % log_every == 0) {
            std::cout<<"Running, iter #"<< i + 1<< " / "<< niter<<std::endl;
        }
    }
    std::cout<<"END!"<<std::endl;
    return 0;

}
