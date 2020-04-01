#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <deque>
#include <string>

#include "conditional_mcmc.hpp"
#include "factory.hpp"
#include "utils.hpp"
#include "../protos/cpp/params.pb.h"


namespace py = pybind11;

std::deque<py::bytes> _run_pp_mix(
        int burnin, int niter, int thin, const Eigen::MatrixXd& data,
        std::string serialized_params, std::vector<double> inits = {},
        bool fixed_pp=false) {

    std::deque<py::bytes> out;
    int log_every = 200;

    Eigen::MatrixXd ranges(2, data.cols());
    ranges.row(0) = data.colwise().minCoeff();
    ranges.row(1) = data.colwise().maxCoeff();
    ranges *= 2;
    std::cout << "ranges: \n" << ranges << std::endl;

    Params params;
    params.ParseFromString(serialized_params);

    BasePP *pp_mix = make_pp(params);
    BaseJump *h = make_jump(params);
    BasePrec *g = make_prec(params);
    pp_mix->set_ranges(ranges);

    ConditionalMCMC sampler(pp_mix, h, g);
    sampler.initialize(data);

    for (int i=0; i < burnin; i++) {
        sampler.run_one();
        if ((i + 1) % log_every == 0) {
            py::print("Burnin, iter #", i + 1, " / ", burnin);
        }
    }

    for (int i = 0; i < niter; i++) {
        sampler.run_one();
        if (i % thin == 0) {
            std::string s;
            sampler.get_state_as_proto().SerializeToString(&s);
            out.push_back((py::bytes)s);
        }

        if ((i + 1) % log_every == 0) {
            py::print("Running, iter #", i + 1, " / ", niter);
        }
    }
    return out;
}


Eigen::MatrixXd _simulate_strauss2D(
    const Eigen::MatrixXd& ranges, double beta, double gamma, double R) {
    return simulate_strauss_moller(ranges, beta, gamma, R);
}


PYBIND11_MODULE(pp_mix_cpp, m)
{
    m.doc() = "aaa"; // optional module docstring

    m.def("_run_pp_mix", &_run_pp_mix, "aaa");

    m.def("_simulate_strauss2D", &_simulate_strauss2D, "aaa");
}
