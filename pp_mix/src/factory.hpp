#ifndef FACTORY_HPP
#define FACTORY_HPP

#include <memory>

#include "precs/base_prec.hpp"
#include "precs/fixed_prec.hpp"
#include "precs/delta_wishart.hpp"
#include "precs/delta_gamma.hpp"

#include "point_process/determinantalPP.hpp"

#include "../protos/cpp/params.pb.h"

// DPP
DeterminantalPP* make_dpp(const Params& params, const MatrixXd& ranges);

// Delta Precision
BasePrec* make_delta(const Params& params);

BasePrec *make_fixed_prec(const FixedMultiPrecParams &params);

BasePrec* make_wishart(const WishartParams& params);

BasePrec* make_fixed_prec(const FixedUnivPrecParams& params);

BasePrec* make_gamma_prec(const GammaParams& params);

#endif
