#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <deque>
#include <string>

#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "mala_conditional_mcmc.hpp"
#include "factory.hpp"
#include "utils.hpp"

namespace py = pybind11;

/////////////////////////// This is the main function provided to Python user

std::deque<py::bytes> _run_pp_mix(int ntrick, int burnin, int niter, int thin,
                                  const Eigen::MatrixXd &data,
                                  std::string serialized_params,
                                  const Eigen::MatrixXd &ranges,
                                  int log_every = 200) {
  Params params;
  params.ParseFromString(serialized_params);


  std::deque<py::bytes> out;
  DeterminantalPP* pp_mix = make_dpp(params, ranges);
  BasePrec* g = make_delta(params);

  if (params.has_MH_sigma())
    Mala::ClassicalMultiMCMC sampler(pp_mix, g, params);
  else if (params.has_mala_step())
    Mala::MalaMultiMCMC sampler(pp_mix, g, params);

  
  sampler.initialize(data);

  for (int i = 0; i < ntrick; i++) {
    sampler.run_one_trick();
    if ((i + 1) % log_every == 0) {
      py::print("Trick, iter #", i + 1, " / ", ntrick);
    }
  }

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



PYBIND11_MODULE(pp_mix_cpp, m) {
  m.doc() = "aaa";  // optional module docstring

  m.def("_run_pp_mix", &_run_pp_mix, "aaa");

}
