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

    int niter=2;

    MatrixXd datacsv = load_csv<MatrixXd>("/home/lorenzo/Documents/Tesi/github_repos/pp_mix/data/data_autodiff_analytic/p_100_n_150.csv");

    std::string params_file = \
      "/home/lorenzo/Documents/Tesi/github_repos/pp_mix/pp_mix/resources/autodiff_analytic/sampler_params_d_vary_N_vary.asciipb";
    Params params = loadTextProto<Params>(params_file);

    // We perform 4 steps updating Lambda, other parameters fixed.
    DeterminantalPP* pp_mix = make_dpp(params);

    BasePrec* g = make_delta(params);

    MCMCsampler::MultivariateConditionalMCMC sampler(pp_mix, g, params);

    sampler.initialize(datacsv);

    for (int i = 0; i < niter; i++) {
        sampler.run_one();

            std::cout<<"Running, iter #"<< i + 1<< " / "<< niter<<std::endl;
    }


    std::cout<<"END!"<<std::endl;
    return 0;

}
