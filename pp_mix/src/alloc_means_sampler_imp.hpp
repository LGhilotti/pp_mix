#ifndef ALLOC_MEANS_SAMPLER_IMP
#define ALLOC_MEANS_SAMPLER_IMP

namespace MCMCsampler {

// returns the ln of full-cond of Lambda|rest in current Lambda (lamb is vectorized)
template <typename T>
T MeansSamplerMala::means_target_function::operator()(const Eigen::Matrix<T,Eigen::Dynamic,1> & lamb) const {

  using std::pow; using std::exp; using std::log;



}

};


#endif