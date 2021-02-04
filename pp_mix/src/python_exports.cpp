//#include <Eigen/Dense>
#include <deque>
#include <string>

#include <stan/math/fwd.hpp>
#include <stan/math/mix.hpp>
#include <stan/math/prim.hpp>
#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
/// COLPA DI QUESTO EIGEN.H
//#include <pybind11/eigen.h>


#include "../protos/cpp/params.pb.h"
#include "../protos/cpp/state.pb.h"
#include "mala_conditional_mcmc.hpp"
#include "factory.hpp"
#include "utils.hpp"

namespace py = pybind11;

/////////////////////////// This is the main function provided to Python user

std::deque<py::bytes> _run_pp_mix(int ntrick, int burnin, int niter, int thin,
                                  std::string serialized_data, //const Eigen::MatrixXd &data,
                                  std::string serialized_params,
                                  std::string serialized_ranges, //const Eigen::MatrixXd &ranges,
                                  int log_every = 200) {
  Params params;
  params.ParseFromString(serialized_params);
  EigenMatrix data_proto;
  data_proto.ParseFromString(serialized_data);
  EigenMatrix ranges_proto;
  ranges_proto.ParseFromString(serialized_ranges);

  Eigen::MatrixXd data = to_eigen(data_proto);
  Eigen::MatrixXd ranges = to_eigen(ranges_proto); 

  std::deque<py::bytes> out;
  DeterminantalPP* pp_mix = make_dpp(params, ranges);
  BasePrec* g = make_delta(params);

  Mala::MultivariateConditionalMCMC* sampler = make_sampler(params, pp_mix, g);
  
  sampler->initialize(data);

  for (int i = 0; i < ntrick; i++) {
    sampler->run_one_trick();
    if ((i + 1) % log_every == 0) {
      py::print("Trick, iter #", i + 1, " / ", ntrick);
    }
  }

  for (int i = 0; i < burnin; i++) {
    sampler->run_one();
    if ((i + 1) % log_every == 0) {
      py::print("Burnin, iter #", i + 1, " / ", burnin);
    }
  }

  for (int i = 0; i < niter; i++) {
    sampler->run_one();
    if (i % thin == 0) {
      std::string s;
      MultivariateMixtureState curr;
      sampler->get_state_as_proto(&curr);
      curr.SerializeToString(&s);
      out.push_back((py::bytes)s);
    }

    if ((i + 1) % log_every == 0) {
      py::print("Running, iter #", i + 1, " / ", niter);
    }
  }

  return out;
}



PYBIND11_MODULE(pp_mix_high, m) {
  m.doc() = "aaa";  // optional module docstring

  m.def("_run_pp_mix", &_run_pp_mix, "aaa");

}
