#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <deque>
#include <string>

#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "conditional_mcmc.hpp"
#include "factory.hpp"
#include "utils.hpp"

namespace py = pybind11;

std::deque<py::bytes> run_pp_mix_univ(int burnin, int niter, int thin,
                                      const Eigen::MatrixXd &data,
                                      Params params, const Eigen::MatrixXd &ranges,
                                      int log_every) {

  std::cout << "ranges: \n" << ranges << std::endl;

  std::deque<py::bytes> out;
  BasePP *pp_mix = make_pp(params);
  BasePrec *g = make_prec(params);
  pp_mix->set_ranges(ranges);

  UnivariateConditionalMCMC sampler(pp_mix, g, params);
  sampler.initialize(data);

  for (int i = 0; i < burnin; i++) {
    sampler.run_one();
    if ((i + 1) % log_every == 0) {
      py::print("Burnin, iter #", i + 1, " / ", burnin);
    }
  }

  for (int i = 0; i < niter; i++) {
    sampler.run_one();
    if (i % thin == 0) {
      std::string s;
      UnivariateMixtureState curr;
      sampler.get_state_as_proto(&curr);
      curr.SerializeToString(&s);
      out.push_back((py::bytes)s);
    }

    if ((i + 1) % log_every == 0) {
      py::print("Running, iter #", i + 1, " / ", niter);
    }
  }
  return out;
}

std::deque<py::bytes> run_pp_mix_multi(int burnin, int niter, int thin,
                                       const Eigen::MatrixXd &data,
                                       Params params, const Eigen::MatrixXd &ranges,
                                       int log_every) {

  std::cout << "ranges: \n" << ranges << std::endl;

  std::deque<py::bytes> out;
  BasePP *pp_mix = make_pp(params);
  BasePrec *g = make_prec(params);
  // make_pp only sets PP-specific parameter members.
  // Now, sets member of BasePP(ranges, dim,...) and calls initialize() which sets PP-specific members (eigen-decomposition,..)
  pp_mix->set_ranges(ranges);

  MultivariateConditionalMCMC sampler(pp_mix, h, g, params);
  sampler.initialize(data);

  for (int i = 0; i < burnin; i++) {
    sampler.run_one();
    if ((i + 1) % log_every == 0) {
      py::print("Burnin, iter #", i + 1, " / ", burnin);
    }
  }

  // sampler.set_verbose();
  for (int i = 0; i < niter; i++) {
    sampler.run_one();
    if (i % thin == 0) {
      std::string s;
      MultivariateMixtureState curr;
      sampler.get_state_as_proto(&curr);
      curr.SerializeToString(&s);
      out.push_back((py::bytes)s);
    }

    if ((i + 1) % log_every == 0) {
      py::print("Running, iter #", i + 1, " / ", niter);
    }
  }
  return out;
}


/////////////////////////// This is the main function provided to Python user

std::deque<py::bytes> _run_pp_mix(int burnin, int niter, int thin,
                                  const Eigen::MatrixXd &data,
                                  std::string serialized_params,
                                  const Eigen::MatrixXd &ranges,
                                  int log_every = 200) {
  Params params;
  params.ParseFromString(serialized_params);

  if (data.rows() == 1 || data.cols() == 1)
    return run_pp_mix_univ(burnin, niter, thin, data, params, ranges, log_every);

  else
    return run_pp_mix_multi(burnin, niter, thin, data, params, ranges, log_every);
}



PYBIND11_MODULE(pp_mix_cpp, m) {
  m.doc() = "aaa";  // optional module docstring

  m.def("_run_pp_mix", &_run_pp_mix, "aaa");

}
